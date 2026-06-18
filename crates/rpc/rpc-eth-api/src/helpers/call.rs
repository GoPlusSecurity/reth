//! Loads a pending block from database. Helper trait for `eth_` transaction, call and trace RPC
//! methods.

use core::fmt;
use std::collections::HashMap;

use super::{LoadBlock, LoadPendingBlock, LoadState, LoadTransaction, SpawnBlocking, Trace};
use crate::{
    helpers::estimate::EstimateCall, FromEvmError, FullEthApiTypes, RpcBlock, RpcNodeCore,
};
use alloy_consensus::{transaction::TxHashRef, BlockHeader};
use alloy_eips::eip2930::AccessListResult;
use alloy_evm::overrides::{apply_block_overrides, apply_state_overrides, OverrideBlockHashes};
use alloy_network::TransactionBuilder;
use alloy_primitives::{address, keccak256, Address, Bytes, B256, U256};
use alloy_rpc_types_eth::{
    simulate::{SimBlock, SimulatePayload, SimulatedBlock},
    state::{EvmOverrides, StateOverride},
    BlockId, Bundle, EthCallResponse, StateContext, TransactionInfo,
};
use futures::Future;
use reth_chainspec::{ChainSpecProvider, EthChainSpec, EthereumHardforks};
use reth_errors::{ProviderError, RethError};
use reth_evm::{
    block::BlockExecutor, env::BlockEnvironment, execute::BlockBuilder, ConfigureEvm, Evm,
    EvmEnvFor, HaltReasonFor, InspectorFor, TransactionEnvMut, TxEnvFor,
};
use reth_node_api::BlockBody;
use reth_primitives_traits::Recovered;
use reth_revm::{
    cancelled::CancelOnDrop,
    database::StateProviderDatabase,
    db::{bal::EvmDatabaseError, State},
};
use reth_rpc_convert::{RpcConvert, RpcTxReq};
use reth_rpc_eth_types::{
    cache::db::StateProviderTraitObjWrapper,
    error::{AsEthApiError, FromEthApiError},
    simulate::{self, EthSimulateError},
    EthApiError, StateCacheDb,
};
use reth_storage_api::{BlockIdReader, ProviderTx, StateProviderBox};
use revm::{
    context::Block,
    context_interface::{result::ResultAndState, Transaction},
    Database, DatabaseCommit,
};
use revm_inspectors::{access_list::AccessListInspector, transfer::TransferInspector};
use std::collections::BTreeMap;
use tracing::{trace, warn};
use serde::{Deserialize, Serialize};

/// Result type for `eth_simulateV1` RPC method.
pub type SimulatedBlocksResult<N, E> = Result<Vec<SimulatedBlock<RpcBlock<N>>>, E>;

const BASE_WETH_ADDRESS: Address = address!("0x4200000000000000000000000000000000000006");

/// Simplified event log shape returned by `eth_callSequence`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CallSequenceLog {
    /// Contract that emitted this log.
    pub address: Address,
    /// Indexed topics emitted by the log.
    pub topics: Vec<B256>,
    /// Raw data payload.
    pub data: Bytes,
}

/// Per-transaction result returned by `eth_callSequence`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CallSequenceResult {
    /// Whether the transaction call executed without halt/revert.
    pub ok: bool,
    /// Return bytes (success output or revert bytes when available).
    pub result: Bytes,
    /// Simplified logs from this transaction.
    pub logs: Vec<CallSequenceLog>,
    /// Gas used by this transaction.
    pub used_gas: u64,
    /// Signed sender balance delta encoded as decimal string.
    pub from_balance_change: String,
    /// Decoded or synthesized revert/halt reason; empty on success.
    pub revert_reason: String,
}

/// Native balance change for one account.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CallSequenceNativeChange {
    /// Signed delta encoded as decimal string.
    pub change: String,
    /// Balance before the call.
    pub before: String,
    /// Balance after the call.
    pub after: String,
}

/// ERC-721 ownership change for one token id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CallSequenceErc721TokenChange {
    /// Token identifier encoded as decimal string.
    pub token_id: String,
    /// `true` if account received ownership, `false` if account lost ownership.
    pub received: bool,
}

/// ERC-1155 balance change for one token id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CallSequenceErc1155TokenChange {
    /// Token identifier encoded as decimal string.
    pub token_id: String,
    /// Signed token delta encoded as decimal string.
    pub change: String,
}

/// Per-transaction result returned by `eth_callSequenceWithBalanceTracking`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CallSequenceWithBalanceTrackingResult {
    /// Whether the transaction call executed without halt/revert.
    pub ok: bool,
    /// Return bytes (success output or revert bytes when available).
    pub result: Bytes,
    /// Simplified logs from this transaction.
    pub logs: Vec<CallSequenceLog>,
    /// Gas used by this transaction.
    pub used_gas: u64,
    /// Native balance changes keyed by account.
    pub native_changes: HashMap<Address, CallSequenceNativeChange>,
    /// ERC-20 deltas keyed by owner account, then token contract.
    pub erc20_changes: HashMap<Address, HashMap<Address, String>>,
    /// ERC-721 ownership changes keyed by owner account, then token contract.
    pub erc721_changes: HashMap<Address, HashMap<Address, Vec<CallSequenceErc721TokenChange>>>,
    /// ERC-1155 deltas keyed by owner account, then token contract.
    pub erc1155_changes: HashMap<Address, HashMap<Address, Vec<CallSequenceErc1155TokenChange>>>,
    /// Decoded or synthesized revert/halt reason; empty on success.
    pub revert_reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct SignedAmount {
    negative: bool,
    magnitude: U256,
}

impl SignedAmount {
    fn add_positive(&mut self, amount: U256) {
        if amount.is_zero() {
            return
        }

        if self.negative {
            if self.magnitude > amount {
                self.magnitude -= amount;
                return
            }
            if self.magnitude == amount {
                self.negative = false;
                self.magnitude = U256::ZERO;
                return
            }
            self.negative = false;
            self.magnitude = amount - self.magnitude;
            return
        }

        self.magnitude += amount;
    }

    fn add_negative(&mut self, amount: U256) {
        if amount.is_zero() {
            return
        }

        if self.negative {
            self.magnitude += amount;
            return
        }

        if self.magnitude > amount {
            self.magnitude -= amount;
            return
        }
        if self.magnitude == amount {
            self.magnitude = U256::ZERO;
            return
        }
        self.negative = true;
        self.magnitude = amount - self.magnitude;
    }

    fn as_string(self) -> String {
        signed_u256_to_hex(self.negative, self.magnitude)
    }

    fn is_zero(self) -> bool {
        self.magnitude.is_zero()
    }
}

fn signed_balance_delta(before: U256, after: U256) -> String {
    if after >= before {
        signed_u256_to_hex(false, after - before)
    } else {
        signed_u256_to_hex(true, before - after)
    }
}

fn u256_to_hex(value: U256) -> String {
    format!("0x{value:x}")
}

fn signed_u256_to_hex(negative: bool, magnitude: U256) -> String {
    if magnitude.is_zero() {
        return String::from("0x0")
    }
    if negative {
        return format!("-0x{magnitude:x}")
    }
    format!("0x{magnitude:x}")
}

fn topic_to_address(topic: &B256) -> Address {
    let bytes = topic.as_slice();
    Address::from_slice(&bytes[12..32])
}

fn u256_from_word(data: &[u8]) -> Option<U256> {
    if data.len() < 32 {
        return None
    }

    let mut word = [0u8; 32];
    word.copy_from_slice(&data[..32]);
    Some(U256::from_be_bytes(word))
}

fn u256_from_topic(topic: &B256) -> U256 {
    let mut word = [0u8; 32];
    word.copy_from_slice(topic.as_slice());
    U256::from_be_bytes(word)
}

fn parse_abi_u256_array(data: &[u8], offset: usize) -> Option<Vec<U256>> {
    if offset + 32 > data.len() {
        return None
    }

    let len = usize::try_from(u256_from_word(&data[offset..])?).ok()?;
    let mut items = Vec::with_capacity(len);
    let mut cursor = offset + 32;

    for _ in 0..len {
        if cursor + 32 > data.len() {
            return None
        }
        items.push(u256_from_word(&data[cursor..])?);
        cursor += 32;
    }

    Some(items)
}

fn parse_transfer_batch(data: &[u8]) -> Option<Vec<(U256, U256)>> {
    if data.len() < 64 {
        return None
    }

    let ids_offset = usize::try_from(u256_from_word(data)?).ok()?;
    let values_offset = usize::try_from(u256_from_word(&data[32..])?).ok()?;
    let ids = parse_abi_u256_array(data, ids_offset)?;
    let values = parse_abi_u256_array(data, values_offset)?;

    Some(ids.into_iter().zip(values).collect())
}

fn add_erc20_change(
    changes: &mut HashMap<Address, HashMap<Address, SignedAmount>>,
    owner: Address,
    token: Address,
    delta: SignedAmount,
) {
    if delta.is_zero() {
        return
    }

    let token_changes = changes.entry(owner).or_default();
    let value = token_changes.entry(token).or_default();
    if delta.negative {
        value.add_negative(delta.magnitude);
    } else {
        value.add_positive(delta.magnitude);
    }
}

fn add_erc1155_change(
    changes: &mut HashMap<Address, HashMap<Address, HashMap<U256, SignedAmount>>>,
    owner: Address,
    token: Address,
    token_id: U256,
    delta: SignedAmount,
) {
    if delta.is_zero() {
        return
    }

    let token_map = changes.entry(owner).or_default();
    let id_map = token_map.entry(token).or_default();
    let value = id_map.entry(token_id).or_default();
    if delta.negative {
        value.add_negative(delta.magnitude);
    } else {
        value.add_positive(delta.magnitude);
    }
}

fn extract_token_changes(
    logs: &[alloy_primitives::Log],
) -> (
    HashMap<Address, HashMap<Address, String>>,
    HashMap<Address, HashMap<Address, Vec<CallSequenceErc721TokenChange>>>,
    HashMap<Address, HashMap<Address, Vec<CallSequenceErc1155TokenChange>>>,
) {
    let transfer_topic = keccak256("Transfer(address,address,uint256)");
    let transfer_single_topic = keccak256("TransferSingle(address,address,address,uint256,uint256)");
    let transfer_batch_topic = keccak256("TransferBatch(address,address,address,uint256[],uint256[])");
    let deposit_topic = keccak256("Deposit(address,uint256)");
    let withdrawal_topic = keccak256("Withdrawal(address,uint256)");

    let zero_address = Address::ZERO;

    let mut erc20_changes: HashMap<Address, HashMap<Address, SignedAmount>> = HashMap::new();
    let mut erc721_changes: HashMap<Address, HashMap<Address, Vec<CallSequenceErc721TokenChange>>> =
        HashMap::new();
    let mut erc1155_changes: HashMap<Address, HashMap<Address, HashMap<U256, SignedAmount>>> =
        HashMap::new();

    for log in logs {
        let topics = log.data.topics();
        if topics.is_empty() {
            continue
        }

        let topic0 = topics[0];
        let data = log.data.data.as_ref();

        if topic0 == transfer_topic {
            if topics.len() == 3 {
                let Some(amount) = u256_from_word(data) else {
                    continue;
                };
                let from = topic_to_address(&topics[1]);
                let to = topic_to_address(&topics[2]);

                if from != zero_address {
                    add_erc20_change(
                        &mut erc20_changes,
                        from,
                        log.address,
                        SignedAmount { negative: true, magnitude: amount },
                    );
                }
                if to != zero_address {
                    add_erc20_change(
                        &mut erc20_changes,
                        to,
                        log.address,
                        SignedAmount { negative: false, magnitude: amount },
                    );
                }
                continue
            }

            if topics.len() == 4 {
                let from = topic_to_address(&topics[1]);
                let to = topic_to_address(&topics[2]);
                let token_id = u256_to_hex(u256_from_topic(&topics[3]));

                if from != zero_address {
                    erc721_changes.entry(from).or_default().entry(log.address).or_default().push(
                        CallSequenceErc721TokenChange { token_id: token_id.clone(), received: false },
                    );
                }
                if to != zero_address {
                    erc721_changes.entry(to).or_default().entry(log.address).or_default().push(
                        CallSequenceErc721TokenChange { token_id, received: true },
                    );
                }
                continue
            }
        }

        if topic0 == transfer_single_topic {
            if topics.len() < 4 {
                continue
            }
            let from = topic_to_address(&topics[2]);
            let to = topic_to_address(&topics[3]);

            let Some(token_id) = u256_from_word(data) else {
                continue;
            };
            let Some(amount) = u256_from_word(&data[32..]) else {
                continue;
            };

            if from != zero_address {
                add_erc1155_change(
                    &mut erc1155_changes,
                    from,
                    log.address,
                    token_id,
                    SignedAmount { negative: true, magnitude: amount },
                );
            }
            if to != zero_address {
                add_erc1155_change(
                    &mut erc1155_changes,
                    to,
                    log.address,
                    token_id,
                    SignedAmount { negative: false, magnitude: amount },
                );
            }
            continue
        }

        if topic0 == transfer_batch_topic {
            if topics.len() < 4 {
                continue
            }
            let from = topic_to_address(&topics[2]);
            let to = topic_to_address(&topics[3]);

            let Some(items) = parse_transfer_batch(data) else {
                continue;
            };

            for (token_id, amount) in items {
                if from != zero_address {
                    add_erc1155_change(
                        &mut erc1155_changes,
                        from,
                        log.address,
                        token_id,
                        SignedAmount { negative: true, magnitude: amount },
                    );
                }
                if to != zero_address {
                    add_erc1155_change(
                        &mut erc1155_changes,
                        to,
                        log.address,
                        token_id,
                        SignedAmount { negative: false, magnitude: amount },
                    );
                }
            }
            continue
        }

        if topic0 == deposit_topic {
            if log.address != BASE_WETH_ADDRESS || topics.len() < 2 {
                continue
            }
            let to = topic_to_address(&topics[1]);
            let Some(amount) = u256_from_word(data) else {
                continue;
            };
            if to != zero_address {
                add_erc20_change(
                    &mut erc20_changes,
                    to,
                    log.address,
                    SignedAmount { negative: false, magnitude: amount },
                );
            }
            continue
        }

        if topic0 == withdrawal_topic {
            if log.address != BASE_WETH_ADDRESS || topics.len() < 2 {
                continue
            }
            let from = topic_to_address(&topics[1]);
            let Some(amount) = u256_from_word(data) else {
                continue;
            };
            if from != zero_address {
                add_erc20_change(
                    &mut erc20_changes,
                    from,
                    log.address,
                    SignedAmount { negative: true, magnitude: amount },
                );
            }
        }
    }

    let mut erc20_out: HashMap<Address, HashMap<Address, String>> = HashMap::new();
    for (owner, by_token) in erc20_changes {
        let mut token_map = HashMap::new();
        for (token, delta) in by_token {
            if !delta.is_zero() {
                token_map.insert(token, delta.as_string());
            }
        }
        if !token_map.is_empty() {
            erc20_out.insert(owner, token_map);
        }
    }

    let mut erc1155_out: HashMap<Address, HashMap<Address, Vec<CallSequenceErc1155TokenChange>>> =
        HashMap::new();
    for (owner, by_token) in erc1155_changes {
        let mut token_map = HashMap::new();
        for (token, by_id) in by_token {
            let mut id_changes = Vec::new();
            for (token_id, delta) in by_id {
                if !delta.is_zero() {
                    id_changes.push(CallSequenceErc1155TokenChange {
                        token_id: u256_to_hex(token_id),
                        change: delta.as_string(),
                    });
                }
            }
            if !id_changes.is_empty() {
                token_map.insert(token, id_changes);
            }
        }
        if !token_map.is_empty() {
            erc1155_out.insert(owner, token_map);
        }
    }

    (erc20_out, erc721_changes, erc1155_out)
}

/// Execution related functions for the [`EthApiServer`](crate::EthApiServer) trait in
/// the `eth_` namespace.
pub trait EthCall: EstimateCall + Call + LoadPendingBlock + LoadBlock + FullEthApiTypes {
    /// Estimate gas needed for execution of the `request` at the [`BlockId`].
    fn estimate_gas_at(
        &self,
        request: RpcTxReq<<Self::RpcConvert as RpcConvert>::Network>,
        at: BlockId,
        overrides: EvmOverrides,
    ) -> impl Future<Output = Result<U256, Self::Error>> + Send {
        EstimateCall::estimate_gas_at(self, request, at, overrides)
    }

    /// `eth_simulateV1` executes an arbitrary number of transactions on top of the requested state.
    /// The transactions are packed into individual blocks. Overrides can be provided.
    ///
    /// See also: <https://github.com/ethereum/go-ethereum/pull/27720>
    fn simulate_v1(
        &self,
        payload: SimulatePayload<RpcTxReq<<Self::RpcConvert as RpcConvert>::Network>>,
        block: Option<BlockId>,
    ) -> impl Future<Output = SimulatedBlocksResult<Self::NetworkTypes, Self::Error>> + Send {
        async move {
            if payload.block_state_calls.len() > self.max_simulate_blocks() as usize {
                return Err(EthApiError::other(EthSimulateError::TooManyBlocks).into())
            }

            let block = block.unwrap_or_default();

            let SimulatePayload {
                block_state_calls,
                trace_transfers,
                validation,
                return_full_transactions,
            } = payload;

            if block_state_calls.is_empty() {
                return Err(EthApiError::InvalidParams(String::from("calls are empty.")).into())
            }

            let _permit = self.acquire_owned_blocking_io().await;

            let base_block =
                self.recovered_block(block).await?.ok_or(EthApiError::HeaderNotFound(block))?;
            let parent = base_block.sealed_header().clone();
            let max_simulate_blocks = self.max_simulate_blocks();

            self.spawn_with_state_at_block(block, move |this, db| {
                let state_provider = db.database.0 .0;
                let mut db = State::builder()
                    .with_database(StateProviderDatabase::new(&state_provider))
                    .with_bundle_update()
                    .build();
                let mut parent = parent;

                let chain_id = this.provider().chain_spec().chain_id();

                // Validate block ordering and fill gaps with empty blocks so every entry has an
                // explicit `number` and `time` override and the chain is contiguous (see the
                // execution-apis spec note: "If the block number is increased more than 1 compared
                // to the previous block, new empty blocks are generated in between.").
                let block_state_calls = simulate::sanitize_chain(
                    block_state_calls,
                    &parent,
                    chain_id,
                    max_simulate_blocks,
                )?;

                let mut blocks: Vec<SimulatedBlock<RpcBlock<Self::NetworkTypes>>> =
                    Vec::with_capacity(block_state_calls.len());

                let call_gas_limit = this.call_gas_limit();
                let mut remaining_call_gas_limit = (call_gas_limit > 0).then_some(call_gas_limit);

                for block in block_state_calls {
                    let SimBlock { block_overrides, state_overrides, calls } = block;

                    let attributes = this
                        .pending_env_builder()
                        .pending_env_attributes(&parent, block_overrides.as_ref())
                        .map_err(Self::Error::from_eth_err)?;

                    let mut evm_env = this
                        .evm_config()
                        .next_evm_env(&parent, &attributes)
                        .map_err(RethError::other)
                        .map_err(Self::Error::from_eth_err)?;

                    // Always disable EIP-3607
                    evm_env.cfg_env.disable_eip3607 = true;

                    if !validation {
                        // If not explicitly required, we disable nonce check <https://github.com/paradigmxyz/reth/issues/16108>
                        evm_env.cfg_env.disable_nonce_check = true;
                        evm_env.cfg_env.disable_base_fee = true;
                        evm_env.cfg_env.tx_gas_limit_cap = Some(u64::MAX);
                        evm_env.block_env.inner_mut().basefee = 0;
                    }

                    // Set prevrandao to zero for simulated blocks by default,
                    // matching spec behavior where MixDigest is zero-initialized.
                    // If user provides an override, it will be applied by apply_block_overrides.
                    evm_env.block_env.inner_mut().prevrandao = Some(B256::ZERO);
                    if !this
                        .provider()
                        .chain_spec()
                        .is_paris_active_at_block(evm_env.block_env.number().saturating_to())
                    {
                        evm_env.block_env.inner_mut().difficulty = parent.difficulty();
                    }

                    if let Some(block_overrides) = block_overrides {
                        // ensure we don't allow uncapped gas limit per block
                        if let Some(gas_limit_override) = block_overrides.gas_limit &&
                            gas_limit_override > evm_env.block_env.gas_limit() &&
                            gas_limit_override > this.call_gas_limit()
                        {
                            return Err(EthApiError::other(EthSimulateError::GasLimitReached).into())
                        }
                        apply_block_overrides(
                            block_overrides,
                            &mut db,
                            evm_env.block_env.inner_mut(),
                        );
                    }
                    if let Some(ref state_overrides) = state_overrides {
                        apply_state_overrides(state_overrides.clone(), &mut db)
                            .map_err(Self::Error::from_eth_err)?;
                    }

                    let chain_id = evm_env.cfg_env.chain_id;

                    let ctx = this
                        .evm_config()
                        .context_for_next_block(&parent, attributes)
                        .map_err(RethError::other)
                        .map_err(Self::Error::from_eth_err)?;
                    let map_err = |e: EthApiError| -> Self::Error {
                        match e.as_simulate_error() {
                            Some(sim_err) => Self::Error::from_eth_err(EthApiError::other(sim_err)),
                            None => Self::Error::from_eth_err(e),
                        }
                    };

                    let (result, results) = if trace_transfers {
                        // prepare inspector to capture transfer inside the evm so they are recorded
                        // and included in logs
                        let inspector = TransferInspector::new(false).with_logs(true);
                        let evm = this
                            .evm_config()
                            .evm_with_env_and_inspector(&mut db, evm_env, inspector);
                        let mut builder = this.evm_config().create_block_builder(evm, &parent, ctx);

                        if let Some(ref state_overrides) = state_overrides {
                            simulate::apply_precompile_overrides(
                                state_overrides,
                                builder.evm_mut().precompiles_mut(),
                            )
                            .map_err(|e| Self::Error::from_eth_err(EthApiError::other(e)))?;
                        }

                        simulate::execute_transactions(
                            builder,
                            &state_provider,
                            calls,
                            &mut remaining_call_gas_limit,
                            chain_id,
                            this.compute_state_root_for_eth_simulate(),
                            this.converter(),
                        )
                        .map_err(map_err)?
                    } else {
                        let evm = this.evm_config().evm_with_env(&mut db, evm_env);
                        let mut builder = this.evm_config().create_block_builder(evm, &parent, ctx);

                        if let Some(ref state_overrides) = state_overrides {
                            simulate::apply_precompile_overrides(
                                state_overrides,
                                builder.evm_mut().precompiles_mut(),
                            )
                            .map_err(|e| Self::Error::from_eth_err(EthApiError::other(e)))?;
                        }

                        simulate::execute_transactions(
                            builder,
                            &state_provider,
                            calls,
                            &mut remaining_call_gas_limit,
                            chain_id,
                            this.compute_state_root_for_eth_simulate(),
                            this.converter(),
                        )
                        .map_err(map_err)?
                    };

                    let simulated_header = result.block.clone_sealed_header();
                    db.override_block_hashes(BTreeMap::from([(
                        simulated_header.number(),
                        simulated_header.hash(),
                    )]));
                    parent = simulated_header;

                    let block = simulate::build_simulated_block::<Self::Error, _>(
                        result.block,
                        results,
                        return_full_transactions.into(),
                        this.converter(),
                    )?;

                    blocks.push(block);
                }

                Ok(blocks)
            })
            .await
        }
    }

    /// Executes the call request (`eth_call`) and returns the output
    fn call(
        &self,
        request: RpcTxReq<<Self::RpcConvert as RpcConvert>::Network>,
        block_number: Option<BlockId>,
        overrides: EvmOverrides,
    ) -> impl Future<Output = Result<Bytes, Self::Error>> + Send {
        async move {
            let _permit = self.acquire_owned_blocking_io().await;
            let res =
                self.transact_call_at(request, block_number.unwrap_or_default(), overrides).await?;

            Self::Error::ensure_success(res.result)
        }
    }

    /// Executes a sequence of calls over the same ephemeral state snapshot.
    ///
    /// Calls are executed in-order and each transaction state transition is committed into the
    /// runtime database so subsequent calls observe prior writes.
    fn call_sequence(
        &self,
        calls: Vec<RpcTxReq<<Self::RpcConvert as RpcConvert>::Network>>,
        block_number: Option<BlockId>,
        mut state_override: Option<StateOverride>,
    ) -> impl Future<Output = Result<Vec<CallSequenceResult>, Self::Error>> + Send {
        async move {
            if calls.is_empty() {
                return Err(EthApiError::InvalidParams(String::from("calls are empty.")).into())
            }

            let mut target_block = block_number.unwrap_or_default();

            if !target_block.is_pending() {
                target_block = self
                    .provider()
                    .block_hash_for_id(target_block)
                    .map_err(|_| EthApiError::HeaderNotFound(target_block))?
                    .ok_or_else(|| EthApiError::HeaderNotFound(target_block))?
                    .into();
            }

            let (evm_env, at) = self.evm_env_at(target_block).await?;

            self.spawn_with_state_at_block(at, move |this, mut db| {
                let mut results = Vec::with_capacity(calls.len());

                for request in calls {
                    let overrides = EvmOverrides::new(state_override.take(), None);
                    let (current_evm_env, prepared_tx) =
                        this.prepare_call_env(evm_env.clone(), request, &mut db, overrides)?;

                    let caller = prepared_tx.caller();

                    let before_balance = db
                        .basic(caller)
                        .map_err(|err: EvmDatabaseError<ProviderError>| {
                            Self::Error::from_eth_err(err)
                        })?
                        .map(|acc| acc.balance)
                        .unwrap_or_default();

                    let res = this.transact(&mut db, current_evm_env, prepared_tx)?;

                    let used_gas = res.result.tx_gas_used();
                    let (ok, result, logs, revert_reason) = match res.result {
                        revm::context_interface::result::ExecutionResult::Success {
                            output,
                            logs,
                            ..
                        } => {
                            let logs = logs
                                .into_iter()
                                .map(|log| CallSequenceLog {
                                    address: log.address,
                                    topics: log.data.topics().to_vec(),
                                    data: log.data.data,
                                })
                                .collect();
                            (true, output.into_data(), logs, String::new())
                        }
                        revm::context_interface::result::ExecutionResult::Revert {
                            output,
                            ..
                        } => {
                            let revert_reason = String::from("execution reverted");
                            (false, output, Vec::new(), revert_reason)
                        }
                        revm::context_interface::result::ExecutionResult::Halt {
                            reason,
                            ..
                        } => {
                            let revert_reason = format!("halt: {reason:?}");
                            (false, Bytes::new(), Vec::new(), revert_reason)
                        }
                    };

                    db.commit(res.state);

                    let after_balance = db
                        .basic(caller)
                        .map_err(|err: EvmDatabaseError<ProviderError>| {
                            Self::Error::from_eth_err(err)
                        })?
                        .map(|acc| acc.balance)
                        .unwrap_or_default();

                    results.push(CallSequenceResult {
                        ok,
                        result,
                        logs,
                        used_gas,
                        from_balance_change: signed_balance_delta(before_balance, after_balance),
                        revert_reason,
                    });
                }

                Ok(results)
            })
            .await
        }
    }

    /// Executes a sequence of calls and returns per-call native/token balance changes.
    fn call_sequence_with_balance_tracking(
        &self,
        calls: Vec<RpcTxReq<<Self::RpcConvert as RpcConvert>::Network>>,
        block_number: Option<BlockId>,
        overrides: EvmOverrides,
    ) -> impl Future<Output = Result<Vec<CallSequenceWithBalanceTrackingResult>, Self::Error>> + Send
    {
        async move {
            if calls.is_empty() {
                return Err(EthApiError::InvalidParams(String::from("calls are empty.")).into())
            }

            let mut target_block = block_number.unwrap_or_default();

            if !target_block.is_pending() {
                target_block = self
                    .provider()
                    .block_hash_for_id(target_block)
                    .map_err(|_| EthApiError::HeaderNotFound(target_block))?
                    .ok_or_else(|| EthApiError::HeaderNotFound(target_block))?
                    .into();
            }

            let (evm_env, at) = self.evm_env_at(target_block).await?;
            let mut state_override = overrides.state;
            let block_override = overrides.block;

            self.spawn_with_state_at_block(at, move |this, mut db| {
                let mut results = Vec::with_capacity(calls.len());

                for request in calls {
                    let current_state_override = state_override.take();
                    let overrides =
                        EvmOverrides::new(current_state_override.clone(), block_override.clone());
                    let (current_evm_env, prepared_tx) =
                        this.prepare_call_env(evm_env.clone(), request, &mut db, overrides)?;

                    let res = if let Some(ref state_overrides) = current_state_override {
                        let mut evm = this.evm_config().evm_with_env(&mut db, current_evm_env);
                        simulate::apply_precompile_overrides(
                            state_overrides,
                            evm.precompiles_mut(),
                        )
                        .map_err(|e| Self::Error::from_eth_err(EthApiError::other(e)))?;
                        evm.transact(prepared_tx).map_err(Self::Error::from_evm_err)?
                    } else {
                        this.transact(&mut db, current_evm_env, prepared_tx)?
                    };

                    let mut native_changes = HashMap::new();
                    for address in res.state.keys().copied() {
                        let before = db
                            .basic(address)
                            .map_err(|err: EvmDatabaseError<ProviderError>| {
                                Self::Error::from_eth_err(err)
                            })?
                            .map(|acc| acc.balance)
                            .unwrap_or_default();

                        let after = res
                            .state
                            .get(&address)
                            .map(|acc| acc.info.balance)
                            .unwrap_or_default();

                        if before != after {
                            native_changes.insert(
                                address,
                                CallSequenceNativeChange {
                                    change: if after >= before {
                                        signed_u256_to_hex(false, after - before)
                                    } else {
                                        signed_u256_to_hex(true, before - after)
                                    },
                                    before: u256_to_hex(before),
                                    after: u256_to_hex(after),
                                },
                            );
                        }
                    }

                    let used_gas = res.result.tx_gas_used();
                    let (ok, result, logs, erc20_changes, erc721_changes, erc1155_changes, revert_reason) =
                        match res.result {
                            revm::context_interface::result::ExecutionResult::Success {
                                output,
                                logs,
                                ..
                            } => {
                                let (erc20_changes, erc721_changes, erc1155_changes) =
                                    extract_token_changes(&logs);
                                let rpc_logs = logs
                                    .into_iter()
                                    .map(|log| CallSequenceLog {
                                        address: log.address,
                                        topics: log.data.topics().to_vec(),
                                        data: log.data.data,
                                    })
                                    .collect();
                                (
                                    true,
                                    output.into_data(),
                                    rpc_logs,
                                    erc20_changes,
                                    erc721_changes,
                                    erc1155_changes,
                                    String::new(),
                                )
                            }
                            revm::context_interface::result::ExecutionResult::Revert {
                                output,
                                ..
                            } => {
                                let revert_reason = String::from("execution reverted");
                                (
                                    false,
                                    output,
                                    Vec::new(),
                                    HashMap::new(),
                                    HashMap::new(),
                                    HashMap::new(),
                                    revert_reason,
                                )
                            }
                            revm::context_interface::result::ExecutionResult::Halt {
                                reason,
                                ..
                            } => {
                                let revert_reason = format!("halt: {reason:?}");
                                (
                                    false,
                                    Bytes::new(),
                                    Vec::new(),
                                    HashMap::new(),
                                    HashMap::new(),
                                    HashMap::new(),
                                    revert_reason,
                                )
                            }
                        };

                    db.commit(res.state);

                    results.push(CallSequenceWithBalanceTrackingResult {
                        ok,
                        result,
                        logs,
                        used_gas,
                        native_changes,
                        erc20_changes,
                        erc721_changes,
                        erc1155_changes,
                        revert_reason,
                    });
                }

                Ok(results)
            })
            .await
        }
    }

    /// Simulate arbitrary number of transactions at an arbitrary blockchain index, with the
    /// optionality of state overrides
    fn call_many(
        &self,
        bundles: Vec<Bundle<RpcTxReq<<Self::RpcConvert as RpcConvert>::Network>>>,
        state_context: Option<StateContext>,
        mut state_override: Option<StateOverride>,
    ) -> impl Future<Output = Result<Vec<Vec<EthCallResponse>>, Self::Error>> + Send {
        async move {
            // Check if the vector of bundles is empty
            if bundles.is_empty() {
                return Err(EthApiError::InvalidParams(String::from("bundles are empty.")).into());
            }

            let _permit = self.acquire_owned_blocking_io().await;

            let StateContext { transaction_index, block_number } =
                state_context.unwrap_or_default();
            let transaction_index = transaction_index.unwrap_or_default();

            let mut target_block = block_number.unwrap_or_default();
            let is_block_target_pending = target_block.is_pending();

            // if it's not pending, we should always use block_hash over block_number to ensure that
            // different provider calls query data related to the same block.
            if !is_block_target_pending {
                let Some(block_hash) = self
                    .provider()
                    .block_hash_for_id(target_block)
                    .map_err(Self::Error::from_eth_err::<ProviderError>)?
                else {
                    return Err(EthApiError::HeaderNotFound(target_block).into())
                };
                target_block = block_hash.into();
            }

            let block = self
                .recovered_block(target_block)
                .await?
                .ok_or(EthApiError::HeaderNotFound(target_block))?;
            let evm_env = self.evm_env_for_header(block.sealed_block().sealed_header())?;

            // we're essentially replaying the transactions in the block here, hence we need the
            // state that points to the beginning of the block, which is the state at
            // the parent block
            let mut at = block.parent_hash();
            let mut replay_block_txs = true;

            let num_txs =
                transaction_index.index().unwrap_or_else(|| block.body().transactions().len());
            // but if all transactions are to be replayed, we can use the state at the block itself,
            // however only if we're not targeting the pending block, because for pending we can't
            // rely on the block's state being available
            if !is_block_target_pending && num_txs == block.body().transactions().len() {
                at = block.hash();
                replay_block_txs = false;
            }

            self.spawn_with_state_at_block(at, move |this, mut db| {
                let mut all_results = Vec::with_capacity(bundles.len());

                if replay_block_txs {
                    let mut executor = RpcNodeCore::evm_config(&this)
                        .executor_for_block(&mut db, block.sealed_block())
                        .map_err(RethError::other)
                        .map_err(Self::Error::from_eth_err)?;
                    executor.apply_pre_execution_changes().map_err(Self::Error::from_eth_err)?;
                    for tx in block.transactions_recovered().take(num_txs) {
                        executor.execute_transaction(tx).map_err(Self::Error::from_eth_err)?;
                    }
                }

                // transact all bundles
                for (bundle_index, bundle) in bundles.into_iter().enumerate() {
                    let Bundle { transactions, block_override } = bundle;
                    if transactions.is_empty() {
                        // Skip empty bundles
                        continue;
                    }

                    let mut bundle_results = Vec::with_capacity(transactions.len());
                    let block_overrides = block_override.map(Box::new);

                    // transact all transactions in the bundle
                    for (tx_index, tx) in transactions.into_iter().enumerate() {
                        // Apply overrides, state overrides are only applied for the first tx in the
                        // request
                        let overrides =
                            EvmOverrides::new(state_override.take(), block_overrides.clone());

                        let (current_evm_env, prepared_tx) = this
                            .prepare_call_env(evm_env.clone(), tx, &mut db, overrides)
                            .map_err(|err| {
                                Self::Error::from_eth_err(EthApiError::call_many_error(
                                    bundle_index,
                                    tx_index,
                                    err.into(),
                                ))
                            })?;
                        let res = this.transact(&mut db, current_evm_env, prepared_tx).map_err(
                            |err| {
                                Self::Error::from_eth_err(EthApiError::call_many_error(
                                    bundle_index,
                                    tx_index,
                                    err.into(),
                                ))
                            },
                        )?;

                        match Self::Error::ensure_success(res.result) {
                            Ok(output) => {
                                bundle_results
                                    .push(EthCallResponse { value: Some(output), error: None });
                            }
                            Err(err) => {
                                bundle_results.push(EthCallResponse {
                                    value: None,
                                    error: Some(err.to_string()),
                                });
                            }
                        }

                        // Commit state changes after each transaction to allow subsequent calls to
                        // see the updates
                        db.commit(res.state);
                    }

                    all_results.push(bundle_results);
                }

                Ok(all_results)
            })
            .await
        }
    }

    /// Creates [`AccessListResult`] for the [`RpcTxReq`] at the given
    /// [`BlockId`], or latest block.
    fn create_access_list_at(
        &self,
        request: RpcTxReq<<Self::RpcConvert as RpcConvert>::Network>,
        block_number: Option<BlockId>,
        state_override: Option<StateOverride>,
    ) -> impl Future<Output = Result<AccessListResult, Self::Error>> + Send
    where
        Self: Trace,
    {
        async move {
            let block_id = block_number.unwrap_or_default();
            let (evm_env, at) = self.evm_env_at(block_id).await?;

            self.spawn_blocking_io_fut(async move |this| {
                this.create_access_list_with(evm_env, at, request, state_override).await
            })
            .await
        }
    }

    /// Creates [`AccessListResult`] for the [`RpcTxReq`] at the given
    /// [`BlockId`].
    fn create_access_list_with(
        &self,
        mut evm_env: EvmEnvFor<Self::Evm>,
        at: BlockId,
        request: RpcTxReq<<Self::RpcConvert as RpcConvert>::Network>,
        state_override: Option<StateOverride>,
    ) -> impl Future<Output = Result<AccessListResult, Self::Error>> + Send
    where
        Self: Trace,
    {
        self.spawn_blocking_io_fut(async move |this| {
            let state = this.state_at_block_id(at).await?;
            let mut db = State::builder().with_database(StateProviderDatabase::new(state)).build();

            if let Some(state_overrides) = state_override {
                apply_state_overrides(state_overrides, &mut db)
                    .map_err(Self::Error::from_eth_err)?;
            }

            // Read fields from request before consuming it in create_txn_env
            let request_has_gas_limit = request.as_ref().gas_limit().is_some();
            let initial = request.as_ref().access_list().cloned().unwrap_or_default();

            let mut tx_env = this.create_txn_env(&evm_env, request, &mut db)?;

            // we want to disable this in eth_createAccessList, since this is common practice used
            // by other node impls and providers <https://github.com/foundry-rs/foundry/issues/4388>
            evm_env.cfg_env.disable_block_gas_limit = true;

            // The basefee should be ignored for eth_createAccessList
            // See:
            // <https://github.com/ethereum/go-ethereum/blob/8990c92aea01ca07801597b00c0d83d4e2d9b811/internal/ethapi/api.go#L1476-L1476>
            evm_env.cfg_env.disable_base_fee = true;

            // Disabled because eth_createAccessList is sometimes used with non-eoa senders
            evm_env.cfg_env.disable_eip3607 = true;

            // Disable additional fee charges (e.g. L2 operator fees),
            // consistent with prepare_call_env and estimate_gas_with.
            evm_env.cfg_env.disable_fee_charge = true;

            // Disable EIP-7825 transaction gas limit cap so that the gas limit
            // fallback (block gas limit) is not rejected when it exceeds the
            // per-tx cap (2^24 ≈ 16.7M post-Osaka).
            evm_env.cfg_env.tx_gas_limit_cap = Some(u64::MAX);

            if !request_has_gas_limit && tx_env.gas_price() > 0 {
                let cap = this.caller_gas_allowance(&mut db, &evm_env, &tx_env)?;
                // no gas limit was provided in the request, so we need to cap the request's gas
                // limit
                tx_env.set_gas_limit(cap.min(evm_env.block_env.gas_limit()));
            }

            let mut inspector = AccessListInspector::new(initial);

            let result = this.inspect(&mut db, evm_env.clone(), tx_env.clone(), &mut inspector)?;
            let access_list = inspector.into_access_list();
            let gas_used = result.result.tx_gas_used();
            tx_env.set_access_list(access_list.clone());
            if let Err(err) = Self::Error::ensure_success(result.result) {
                return Ok(AccessListResult {
                    access_list,
                    gas_used: U256::from(gas_used),
                    error: Some(err.to_string()),
                });
            }

            // transact again to get the exact gas used
            let result = this.transact(&mut db, evm_env, tx_env)?;
            let gas_used = result.result.tx_gas_used();
            let error = Self::Error::ensure_success(result.result).err().map(|e| e.to_string());

            Ok(AccessListResult { access_list, gas_used: U256::from(gas_used), error })
        })
    }
}

/// Executes code on state.
pub trait Call:
    LoadState<
        RpcConvert: RpcConvert<Evm = Self::Evm>,
        Error: FromEvmError<Self::Evm>
                   + From<<Self::RpcConvert as RpcConvert>::Error>
                   + From<ProviderError>,
    > + SpawnBlocking
{
    /// Returns default gas limit to use for `eth_call` and tracing RPC methods.
    ///
    /// Data access in default trait method implementations.
    fn call_gas_limit(&self) -> u64;

    /// Returns the maximum number of blocks accepted for `eth_simulateV1`.
    fn max_simulate_blocks(&self) -> u64;

    /// Returns whether `eth_simulateV1` should compute state roots.
    fn compute_state_root_for_eth_simulate(&self) -> bool;

    /// Returns the maximum memory the EVM can allocate per RPC request.
    fn evm_memory_limit(&self) -> u64;

    /// Returns the max gas limit that the caller can afford given a transaction environment.
    fn caller_gas_allowance(
        &self,
        mut db: impl Database<Error: Into<EthApiError>>,
        _evm_env: &EvmEnvFor<Self::Evm>,
        tx_env: &TxEnvFor<Self::Evm>,
    ) -> Result<u64, Self::Error> {
        alloy_evm::call::caller_gas_allowance(&mut db, tx_env).map_err(Self::Error::from_eth_err)
    }

    /// Executes the closure with the state that corresponds to the given [`BlockId`].
    fn with_state_at_block<F, R>(
        &self,
        at: BlockId,
        f: F,
    ) -> impl Future<Output = Result<R, Self::Error>> + Send
    where
        R: Send + 'static,
        F: FnOnce(Self, StateProviderBox) -> Result<R, Self::Error> + Send + 'static,
    {
        self.spawn_blocking_io_fut(async move |this| {
            let state = this.state_at_block_id(at).await?;
            f(this, state)
        })
    }

    /// Executes the `TxEnv` against the given [Database] without committing state
    /// changes.
    fn transact<DB>(
        &self,
        db: DB,
        evm_env: EvmEnvFor<Self::Evm>,
        tx_env: TxEnvFor<Self::Evm>,
    ) -> Result<ResultAndState<HaltReasonFor<Self::Evm>>, Self::Error>
    where
        DB: Database<Error = EvmDatabaseError<ProviderError>> + fmt::Debug,
    {
        let mut evm = self.evm_config().evm_with_env(db, evm_env);
        let res = evm.transact(tx_env).map_err(Self::Error::from_evm_err)?;

        Ok(res)
    }

    /// Executes the [`reth_evm::EvmEnv`] against the given [Database] without committing state
    /// changes.
    fn transact_with_inspector<DB, I>(
        &self,
        db: DB,
        evm_env: EvmEnvFor<Self::Evm>,
        tx_env: TxEnvFor<Self::Evm>,
        inspector: I,
    ) -> Result<ResultAndState<HaltReasonFor<Self::Evm>>, Self::Error>
    where
        DB: Database<Error = EvmDatabaseError<ProviderError>> + fmt::Debug,
        I: InspectorFor<Self::Evm, DB>,
    {
        let mut evm = self.evm_config().evm_with_env_and_inspector(db, evm_env, inspector);
        let res = evm.transact(tx_env).map_err(Self::Error::from_evm_err)?;

        Ok(res)
    }

    /// Executes the call request at the given [`BlockId`].
    ///
    /// This spawns a new task that obtains the state for the given [`BlockId`] and then transacts
    /// the call [`Self::transact`]. If the future is dropped before the (blocking) transact
    /// call is invoked, then the task is cancelled early, (for example if the request is terminated
    /// early client-side).
    fn transact_call_at(
        &self,
        request: RpcTxReq<<Self::RpcConvert as RpcConvert>::Network>,
        at: BlockId,
        overrides: EvmOverrides,
    ) -> impl Future<Output = Result<ResultAndState<HaltReasonFor<Self::Evm>>, Self::Error>> + Send
    where
        Self: LoadPendingBlock,
    {
        async move {
            let guard = CancelOnDrop::default();
            let cancel = guard.clone();
            let this = self.clone();

            let res = self
                .spawn_with_call_at(request, at, overrides, move |db, evm_env, tx_env| {
                    if cancel.is_cancelled() {
                        // callsite dropped the guard
                        return Err(EthApiError::InternalEthError.into())
                    }
                    this.transact(db, evm_env, tx_env)
                })
                .await;
            drop(guard);
            res
        }
    }

    /// Executes the closure with the state that corresponds to the given [`BlockId`] on a new task
    fn spawn_with_state_at_block<F, R>(
        &self,
        at: impl Into<BlockId>,
        f: F,
    ) -> impl Future<Output = Result<R, Self::Error>> + Send
    where
        F: FnOnce(Self, StateCacheDb) -> Result<R, Self::Error> + Send + 'static,
        R: Send + 'static,
    {
        let at = at.into();
        self.spawn_blocking_io_fut(async move |this| {
            let state = this.state_at_block_id(at).await?;
            let db = State::builder()
                .with_database(StateProviderDatabase::new(StateProviderTraitObjWrapper(state)))
                .build();
            f(this, db)
        })
    }

    /// Prepares the state and env for the given [`RpcTxReq`] at the given [`BlockId`] and
    /// executes the closure on a new task returning the result of the closure.
    ///
    /// This returns the configured [`reth_evm::EvmEnv`] for the given [`RpcTxReq`] at
    /// the given [`BlockId`] and with configured call settings: `prepare_call_env`.
    ///
    /// This is primarily used by `eth_call`.
    ///
    /// # Blocking behaviour
    ///
    /// This assumes executing the call is relatively more expensive on IO than CPU because it
    /// transacts a single transaction on an empty in memory database. Because `eth_call`s are
    /// usually allowed to consume a lot of gas, this also allows a lot of memory operations so
    /// we assume this is not primarily CPU bound and instead spawn the call on a regular tokio task
    /// instead, where blocking IO is less problematic.
    fn spawn_with_call_at<F, R>(
        &self,
        request: RpcTxReq<<Self::RpcConvert as RpcConvert>::Network>,
        at: BlockId,
        overrides: EvmOverrides,
        f: F,
    ) -> impl Future<Output = Result<R, Self::Error>> + Send
    where
        Self: LoadPendingBlock,
        F: FnOnce(
                &mut StateCacheDb,
                EvmEnvFor<Self::Evm>,
                TxEnvFor<Self::Evm>,
            ) -> Result<R, Self::Error>
            + Send
            + 'static,
        R: Send + 'static,
    {
        async move {
            let (evm_env, at) = self.evm_env_at(at).await?;
            self.spawn_with_state_at_block(at, move |this, mut db| {
                let (evm_env, tx_env) =
                    this.prepare_call_env(evm_env, request, &mut db, overrides)?;

                f(&mut db, evm_env, tx_env)
            })
            .await
        }
    }

    /// Retrieves the transaction if it exists and executes it.
    ///
    /// Before the transaction is executed, all previous transaction in the block are applied to the
    /// state by executing them first.
    /// The callback `f` is invoked with the [`ResultAndState`] after the transaction was executed
    /// and the database that points to the beginning of the transaction.
    ///
    /// Note: Implementers should use a threadpool where blocking is allowed, such as
    /// [`BlockingTaskPool`](reth_tasks::pool::BlockingTaskPool).
    fn spawn_replay_transaction<F, R>(
        &self,
        hash: B256,
        f: F,
    ) -> impl Future<Output = Result<Option<R>, Self::Error>> + Send
    where
        Self: LoadBlock + LoadTransaction,
        F: FnOnce(
                TransactionInfo,
                ResultAndState<HaltReasonFor<Self::Evm>>,
                StateCacheDb,
            ) -> Result<R, Self::Error>
            + Send
            + 'static,
        R: Send + 'static,
    {
        async move {
            let (transaction, block) = match self.transaction_and_block(hash).await? {
                None => return Ok(None),
                Some(res) => res,
            };
            let (tx, tx_info) = transaction.split();

            // we need to get the state of the parent block because we're essentially replaying the
            // block the transaction is included in
            let parent_block = block.parent_hash();

            self.spawn_with_state_at_block(parent_block, move |this, mut db| {
                let block_txs = block.transactions_recovered();

                let mut executor = RpcNodeCore::evm_config(&this)
                    .executor_for_block(&mut db, block.sealed_block())
                    .map_err(RethError::other)
                    .map_err(Self::Error::from_eth_err)?;
                executor.apply_pre_execution_changes().map_err(Self::Error::from_eth_err)?;

                // replay all transactions prior to the targeted transaction
                for block_tx in block_txs {
                    if block_tx.tx_hash() == tx.tx_hash() {
                        break;
                    }
                    executor.execute_transaction(block_tx).map_err(Self::Error::from_eth_err)?;
                }

                let tx_env = RpcNodeCore::evm_config(&this).tx_env(tx);

                let res = executor.evm_mut().transact(tx_env).map_err(Self::Error::from_evm_err)?;
                drop(executor);
                f(tx_info, res, db)
            })
            .await
            .map(Some)
        }
    }

    /// Replays all the transactions until the target transaction is found.
    ///
    /// All transactions before the target transaction are executed and their changes are written to
    /// the _runtime_ db ([`State`]).
    ///
    /// Note: This assumes the target transaction is in the given iterator.
    /// Returns the index of the target transaction in the given iterator.
    fn replay_transactions_until<'a, DB, I>(
        &self,
        db: &mut DB,
        evm_env: EvmEnvFor<Self::Evm>,
        transactions: I,
        target_tx_hash: B256,
    ) -> Result<usize, Self::Error>
    where
        DB: Database<Error = EvmDatabaseError<ProviderError>> + DatabaseCommit + core::fmt::Debug,
        I: IntoIterator<Item = Recovered<&'a ProviderTx<Self::Provider>>>,
    {
        let mut evm = self.evm_config().evm_with_env(db, evm_env);
        let mut index = 0;
        for tx in transactions {
            if *tx.tx_hash() == target_tx_hash {
                // reached the target transaction
                break
            }

            let tx_env = self.evm_config().tx_env(tx);
            evm.transact_commit(tx_env).map_err(Self::Error::from_evm_err)?;
            index += 1;
        }
        Ok(index)
    }

    ///
    /// All `TxEnv` fields are derived from the given [`RpcTxReq`], if fields are
    /// `None`, they fall back to the [`reth_evm::EvmEnv`]'s settings.
    fn create_txn_env(
        &self,
        evm_env: &EvmEnvFor<Self::Evm>,
        mut request: RpcTxReq<<Self::RpcConvert as RpcConvert>::Network>,
        mut db: impl Database<Error: Into<EthApiError>>,
    ) -> Result<TxEnvFor<Self::Evm>, Self::Error> {
        if request.as_ref().nonce().is_none() {
            let nonce = db
                .basic(request.as_ref().from().unwrap_or_default())
                .map_err(Into::into)?
                .map(|acc| acc.nonce)
                .unwrap_or_default();
            request.as_mut().set_nonce(nonce);
        }

        Ok(self.converter().tx_env(request, evm_env)?)
    }

    /// Prepares the [`reth_evm::EvmEnv`] for execution of calls.
    ///
    /// Does not commit any changes to the underlying database.
    ///
    /// ## EVM settings
    ///
    /// This modifies certain EVM settings to mirror geth's `SkipAccountChecks` when transacting requests, see also: <https://github.com/ethereum/go-ethereum/blob/380688c636a654becc8f114438c2a5d93d2db032/core/state_transition.go#L145-L148>:
    ///
    ///  - `disable_eip3607` is set to `true`
    ///  - `disable_base_fee` is set to `true`
    ///  - `nonce` is set to `None`
    ///
    /// In addition, this changes the block's gas limit to the configured [`Self::call_gas_limit`].
    #[expect(clippy::type_complexity)]
    fn prepare_call_env<DB>(
        &self,
        mut evm_env: EvmEnvFor<Self::Evm>,
        mut request: RpcTxReq<<Self::RpcConvert as RpcConvert>::Network>,
        db: &mut DB,
        overrides: EvmOverrides,
    ) -> Result<(EvmEnvFor<Self::Evm>, TxEnvFor<Self::Evm>), Self::Error>
    where
        DB: Database + DatabaseCommit + OverrideBlockHashes,
        EthApiError: From<<DB as Database>::Error>,
    {
        // track whether the request has a gas limit set
        let request_has_gas_limit = request.as_ref().gas_limit().is_some();

        if let Some(requested_gas) = request.as_ref().gas_limit() {
            let global_gas_cap = self.call_gas_limit();
            if global_gas_cap != 0 && global_gas_cap < requested_gas {
                warn!(target: "rpc::eth::call", ?request, ?global_gas_cap, "Capping gas limit to global gas cap");
                request.as_mut().set_gas_limit(global_gas_cap);
            }
        } else {
            // cap request's gas limit to call gas limit
            request.as_mut().set_gas_limit(self.call_gas_limit());
        }

        // Disable block gas limit check to allow executing transactions with higher gas limit (call
        // gas limit): https://github.com/paradigmxyz/reth/issues/18577
        evm_env.cfg_env.disable_block_gas_limit = true;

        // Disabled because eth_call is sometimes used with eoa senders
        // See <https://github.com/paradigmxyz/reth/issues/1959>
        evm_env.cfg_env.disable_eip3607 = true;

        // The basefee should be ignored for eth_call
        // See:
        // <https://github.com/ethereum/go-ethereum/blob/ee8e83fa5f6cb261dad2ed0a7bbcde4930c41e6c/internal/ethapi/api.go#L985>
        evm_env.cfg_env.disable_base_fee = true;

        // Disable EIP-7825 transaction gas limit to support larger transactions
        evm_env.cfg_env.tx_gas_limit_cap = Some(u64::MAX);

        // Disable additional fee charges, e.g. opstack operator fee charge
        // See:
        // <https://github.com/paradigmxyz/reth/issues/18470>
        evm_env.cfg_env.disable_fee_charge = true;

        evm_env.cfg_env.memory_limit = self.evm_memory_limit();

        // set nonce to None so that the correct nonce is chosen by the EVM
        request.as_mut().take_nonce();

        if let Some(block_overrides) = overrides.block {
            apply_block_overrides(*block_overrides, db, evm_env.block_env.inner_mut());
        }
        if let Some(state_overrides) = overrides.state {
            apply_state_overrides(state_overrides, db)
                .map_err(EthApiError::from_state_overrides_err)?;
        }

        let mut tx_env = self.create_txn_env(&evm_env, request, &mut *db)?;

        // lower the basefee to 0 to avoid breaking EVM invariants (basefee < gasprice): <https://github.com/ethereum/go-ethereum/blob/355228b011ef9a85ebc0f21e7196f892038d49f0/internal/ethapi/api.go#L700-L704>
        if tx_env.gas_price() == 0 {
            evm_env.block_env.inner_mut().basefee = 0;
        }

        if !request_has_gas_limit {
            // No gas limit was provided in the request, so we need to cap the transaction gas limit
            if tx_env.gas_price() > 0 {
                // If gas price is specified, cap transaction gas limit with caller allowance
                trace!(target: "rpc::eth::call", ?tx_env, "Applying gas limit cap with caller allowance");
                let cap = self.caller_gas_allowance(db, &evm_env, &tx_env)?;
                // ensure we cap gas_limit to the block's
                tx_env.set_gas_limit(cap.min(evm_env.block_env.gas_limit()));
            }
        }

        Ok((evm_env, tx_env))
    }
}
