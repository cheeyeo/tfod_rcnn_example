### Issues

* The ecs instance profile may already exists so need to delete it if it does:

	```
	aws --profile <PROFILE_NAME> iam list-instance-profiles

	aws --profile <PROFILE_NAME> iam delete-instance-profile --instance-profile tfod-ecsInstanceProfile
	```

### Ref

[EC2 instance types]: https://aws.amazon.com/ec2/instance-types/

[TaskDef docs]: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task_definition_parameters.html