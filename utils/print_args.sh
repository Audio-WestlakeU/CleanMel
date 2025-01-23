#!/usr/bin/env bash
###
 # @Author: FnoY fangying@westlake.edu.cn
 # @LastEditors: FnoY0723 fangying@westlake.edu.cn
 # @LastEditTime: 2024-10-10 17:14:03
 # @FilePath: /InASR/utils/print_args.sh
### 
set -euo pipefail

new_args=""
for arg in "${@}"; do
    if [[ ${arg} = *"'"* ]]; then
        arg=$(echo "${arg}" | sed -e "s/'/'\\\\''/g")
    fi

    surround=false
    if [[ ${arg} = *\** ]]; then
        surround=true
    elif [[ ${arg} = *\?* ]]; then
        surround=true
    elif [[ ${arg} = *\\* ]]; then
        surround=true
    elif [[ ${arg} = *\ * ]]; then
        surround=true
    elif [[ ${arg} = *\;* ]]; then
        surround=true
    elif [[ ${arg} = *\&* ]]; then
        surround=true
    elif [[ ${arg} = *\|* ]]; then
        surround=true
    elif [[ ${arg} = *\<* ]]; then
        surround=true
    elif [[ ${arg} = *\>* ]]; then
        surround=true
    elif [[ ${arg} = *\`* ]]; then
        surround=true
    elif [[ ${arg} = *\(* ]]; then
        surround=true
    elif [[ ${arg} = *\)* ]]; then
        surround=true
    elif [[ ${arg} = *\{* ]]; then
        surround=true
    elif [[ ${arg} = *\}* ]]; then
        surround=true
    elif [[ ${arg} = *\[* ]]; then
        surround=true
    elif [[ ${arg} = *\]* ]]; then
        surround=true
    elif [[ ${arg} = *\"* ]]; then
        surround=true
    elif [[ ${arg} = *\#* ]]; then
        surround=true
    elif [[ ${arg} = *\$* ]]; then
        surround=true
    elif [ -z "${arg}" ]; then
        surround=true
    fi

    if "${surround}"; then
        new_args+="'${arg}' "
    else
        new_args+="${arg} "
    fi
done
echo ${new_args}
