#!/bin/bash

set -x

# Script for running one off ECS task via aws cli and jq
config_file=$1

if [[ ! -f ${config_file} ]]; then
	echo "Config file does not exist!"
	exit 1
fi

# Parse config file using jq
profile=$(cat ${config_file} | jq -r '.profile.value')
region=$(cat ${config_file} | jq -r '.region.value')
cluster_name=$(cat ${config_file} | jq -r '.cluster_name.value')
task_definition=$(cat ${config_file} | jq -r '.task_definition.value')

echo "PROFILE: ${profile}"
echo "REGION: ${region}"
echo "CLUSTER NAME: ${cluster_name}"
echo "TASK DEF: ${task_definition}"

TASK_ARN=$(aws ecs run-task --cluster ${cluster_name} --task-definition ${task_definition} --profile ${profile} --region ${region} | jq -r '.tasks[].taskArn')

# NOTE: Below polls every 6 secs; fails after 100 tries
# 6 * 100 = 600s i.e. 10 mins
echo "Watching task: ${TASK_ARN}"
aws ecs wait tasks-running --cluster ${cluster_name} --tasks "${TASK_ARN}" --region ${region} --profile ${profile}
aws ecs wait tasks-stopped --cluster ${cluster_name} --tasks "${TASK_ARN}" --region ${region} --profile ${profile}

# How to log?
# aws --profile ${profile} --region ${region} logs tail /ecs/tfod --log-stream-names "ecs/tfod/d523070853f44e75b67afc526161dea5" --follow --format short