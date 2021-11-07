### Issues

* The ecs instance profile may already exists so need to delete it if it does:

	```
	aws --profile <PROFILE_NAME> iam list-instance-profiles

	aws --profile <PROFILE_NAME> iam delete-instance-profile --instance-profile tfod-ecsInstanceProfile
	```

* Num of GPUS have to match the batch_size in config file else it fails with:

	```
	ValueError: The `global_batch_size` 1 is not divisible by `num_replicas_in_sync` 4 
	```


### Ref

[EC2 instance types]: https://aws.amazon.com/ec2/instance-types/

[TaskDef docs]: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task_definition_parameters.html