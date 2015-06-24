#!/bin/sh
export LOGPATH=${LOCALDIR}/log
mkdir -p ${LOGPATH}
export RUN_DIR=${PWD}
export LAL_SRC=${LOCALDIR}/git/lalsuite
RA=223.0
DEC=-28.5
SKY_ERROR=0
GRB_NAME=100928A
GRB_TIME=969675608
LOCAL_CONFIG_FILES="main.ini postprocessing.ini injections.ini"
BANK_FILE=${PWD}/TMPLTBANKS/H1-TMPLTBANK_GRB100928A_DATAFIND-969673046-4992.xml
./pygrb_make_offline_workflow \
--local-config-files ${LOCAL_CONFIG_FILES} \
--config-overrides \
workflow:ra:${RA} \
workflow:dec:${DEC} \
workflow:sky-error:${SKY_ERROR} \
workflow:trigger-name:${GRB_NAME} \
workflow:trigger-time:${GRB_TIME} \
workflow:start-time:$(( GRB_TIME - 5096 )) \
workflow:end-time:$(( GRB_TIME + 5096 )) \
workflow-tmpltbank:tmpltbank-pregenerated-bank:${BANK_FILE}
