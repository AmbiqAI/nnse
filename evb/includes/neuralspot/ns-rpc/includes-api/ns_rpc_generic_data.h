/**
 * @file ns_rpc_generic_data.h
 * @author Carlos Morales
 * @brief API for using generic data RPC interface
 * @version 0.1
 * @date 2022-08-26
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef NS_RPC_GENERIC_DATA_H
#define NS_RPC_GENERIC_DATA_H

#ifdef __cplusplus
extern "C" {
#endif
#include "GenericDataOperations_EvbToPc.h"
#include "GenericDataOperations_PcToEvb.h"
#include "erpc_server_setup.h"

typedef status (*ns_rpc_data_sendBlockToEVB_cb)(const dataBlock *block);

typedef status (*ns_rpc_data_fetchBlockFromEVB_cb)(dataBlock *block);

typedef status (*ns_rpc_data_computeOnEVB_cb)(const dataBlock *in_block, dataBlock *result_block);

typedef enum { NS_RPC_GENERICDATA_CLIENT, NS_RPC_GENERICDATA_SERVER } rpcGenericDataMode_e;

typedef struct {
    rpcGenericDataMode_e mode;
    ns_rpc_data_sendBlockToEVB_cb sendBlockToEVB_cb;
    ns_rpc_data_fetchBlockFromEVB_cb fetchBlockFromEVB_cb;
    ns_rpc_data_computeOnEVB_cb computeOnEVB_cb;
} ns_rpc_config_t;

/**
 * @brief Initialize the GenericDataOperations RPC system
 *
 * @param cfg
 * @return uint16_t Status
 */
extern uint16_t
ns_rpc_genericDataOperations_init(ns_rpc_config_t *cfg);

extern uint16_t
ns_rpc_genericDataOperationsClient_reset(ns_rpc_config_t *cfg);

/**
 * @brief Helper function for printing a block's contents
 *
 * @param block to be printed
 */
extern void
ns_rpc_genericDataOperations_printDatablock(const dataBlock *block);

/**
 * @brief Helper function to be called after processing the block from ns_rpc_data_fetchBlockFromPC
 * or result block from ns_rpc_data_computeOnPC. This will free() the description
 * and buffer.data block struct members, so only do it after you're done with those!
 *
 * @param block Block to be freed
 */
extern void
ns_rpc_data_clientDoneWithBlockFromPC(const dataBlock *block);

#ifdef __cplusplus
}
#endif

#endif // NS_RPC_GENERIC_DATA_H
