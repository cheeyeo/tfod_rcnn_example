terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

provider "aws" {
  shared_credentials_file = var.aws_credentials
  region                  = var.aws_region
  profile                 = var.aws_profile
}

locals {
  account_id = data.aws_caller_identity.current.account_id
}

# Creating a keypair
resource "aws_key_pair" "deployer" {
  key_name   = var.m1l0_keyname
  public_key = file("~/.ssh/${var.m1l0_keyname}.pub")
}

resource "aws_cloudwatch_log_group" "main" {
  name = "/ecs/tfod"
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "2.64.0"

  cidr = var.vpc_cidr_block

  azs = data.aws_availability_zones.available.names

  private_subnets = slice(var.private_subnet_cidr_blocks, 0, var.private_subnet_count)

  public_subnets = slice(var.public_subnet_cidr_blocks, 0, var.public_subnet_count)

  enable_nat_gateway = true

  enable_vpn_gateway = false
}

module "ssh_security_group" {
  source  = "terraform-aws-modules/security-group/aws"
  version = "3.18.0"

  name        = "SSHDMZ"
  description = "Security group that allows SSH access into VPC"

  vpc_id = module.vpc.vpc_id

  ingress_with_cidr_blocks = [
    {
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = "0.0.0.0/0"
    }
  ]

  egress_with_cidr_blocks = [
    {
      from_port   = 0
      to_port     = 0
      protocol    = "-1"
      cidr_blocks = "0.0.0.0/0"
    }
  ]
}


# Create security group 
module "ssh_private_vpc" {
  source      = "terraform-aws-modules/security-group/aws"
  name        = "PRIVATEVPCSSH"
  description = "Allow SSH access into private VPC"
  vpc_id      = module.vpc.vpc_id

  computed_ingress_with_source_security_group_id = [
    {
      from_port                = 22
      to_port                  = 22
      protocol                 = "tcp"
      source_security_group_id = module.ssh_security_group.this_security_group_id
    }
  ]

  number_of_computed_ingress_with_source_security_group_id = 1

  egress_with_cidr_blocks = [
    {
      from_port   = 0
      to_port     = 0
      protocol    = "-1"
      cidr_blocks = "0.0.0.0/0"
    }
  ]
}

# Creates bastion host
module "bastion_host" {
  source = "./bastion"

  instance_count = 1

  instance_type = "t2.micro"

  subnet_ids = module.vpc.public_subnets

  security_group_ids = [module.ssh_security_group.this_security_group_id]

  key_name = aws_key_pair.deployer.key_name

  tags = {
    Name = "bastion_host"
  }
}



# Creates ECS Cluster
module "ecs" {
  source  = "terraform-aws-modules/ecs/aws"
  version = "2.8.0"

  name = "m1l0"

  tags = {
    Name = "dev"
  }
}

# IAM Roles
resource "aws_iam_role" "ecs_instance_role" {
  name = "${var.tfod_service_name}-ecsInstanceRole"

  description = "IAM Role for ECS Container Instance"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          Service = "ec2.amazonaws.com"
        },
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_instance_ec2_instance_role" {
  role       = aws_iam_role.ecs_instance_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

# create ec2 instance profile
resource "aws_iam_instance_profile" "ecs_instance_profile" {
  name = "${var.tfod_service_name}-ecsInstanceProfile"
  role = aws_iam_role.ecs_instance_role.name
}

# create ecs agent task execution role
resource "aws_iam_role" "ecs_task_role" {
  name = "${var.tfod_service_name}-ecsTaskRole"

  description = "IAM Role for running ECS Agent"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          #AWS     = data.aws_caller_identity.current.arn,
          Service = "ecs-tasks.amazonaws.com"
        },
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "task_role_attach_ecs_policy" {
  role       = aws_iam_role.ecs_task_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

## Service ECS Task role
resource "aws_iam_role" "service_ecs_task_role" {
  name = "${var.tfod_service_name}-ecsServiceTaskRole"

  description = "IAM Role for ECS Service"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        },
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_policy" "service_ecs_task_role_policy" {
  name        = "${var.tfod_service_name}-ecrPolicy"
  description = "Policy for allowing ECR image builds"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchGetImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:PutImage"
        ],
        Resource = "*"
      },
      {
        Effect   = "Allow",
        Action   = "s3:*"
        Resource = "*"
      },
      {
        Effect = "Allow",
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams",
          "logs:PutLogEvents",
          "logs:GetLogEvents",
          "logs:FilterLogEvents"
        ],
        Resource = "*"
      },
      {
        Effect = "Allow",
        Action = [
          "secretsmanager:GetSecretValue"
        ],
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "service_ecs_task_role" {
  role       = aws_iam_role.service_ecs_task_role.name
  policy_arn = aws_iam_policy.service_ecs_task_role_policy.arn
}

## Provision EC2 Instance first
resource "aws_instance" "ecs_instance" {
  ami = data.aws_ami.amazon_ecs_linux_gpu.id

  iam_instance_profile = "${var.tfod_service_name}-ecsInstanceProfile"

  instance_type = var.instance_type

  instance_initiated_shutdown_behavior = "terminate"

  monitoring = false

  security_groups = [module.vpc.default_security_group_id, module.ssh_private_vpc.security_group_id]

  subnet_id = module.vpc.private_subnets[0]

  user_data = data.template_cloudinit_config.config.rendered

  key_name = aws_key_pair.deployer.key_name

  tags = {
    Name = "builder"
  }
}

# Task definition for service
resource "aws_ecs_task_definition" "tfod_service" {
  family                   = var.tfod_service_name
  network_mode             = "bridge"
  requires_compatibilities = ["EC2"]
  cpu                      = var.cpu
  memory                   = var.memory
  task_role_arn            = aws_iam_role.service_ecs_task_role.arn
  execution_role_arn       = aws_iam_role.ecs_task_role.arn

  volume {
    name = "workdir"
    docker_volume_configuration {
      autoprovision = true
      driver        = "local"
      scope         = "shared"
    }
  }

  container_definitions = jsonencode([
    {
      "essential" : true,
      "image" : "${var.container_image}",
      "name" : "${var.tfod_service_name}",
      "resourceRequirements" : [
        {
          "type" : "GPU",
          "value" : "${var.gpus}"
        }
      ],
      "command" : [
          "models",
          "experiments/training",
          "experiments/exported_model",
          "faster_rcnn_lisa_docker.config",
          "s3://tfod"
      ],
      "logConfiguration" : {
        "logDriver" : "awslogs",
        "options" : {
          "awslogs-region" : "${var.aws_region}",
          "awslogs-group" : aws_cloudwatch_log_group.main.name,
          "awslogs-stream-prefix" : "ecs"
        }
      },
      "mountPoints" : [
        {
          "sourceVolume" : "workdir",
          "containerPath" : "/opt/tfod",
          "readOnly" : false
        }
      ]
    }
  ])
}

resource "aws_ecs_service" "tfod_service" {
  name                  = var.tfod_service_name
  cluster               = module.ecs.this_ecs_cluster_name
  task_definition       = aws_ecs_task_definition.tfod_service.arn
  desired_count         = 1
  depends_on            = [aws_instance.ecs_instance]
  wait_for_steady_state = true
  force_new_deployment  = true

  launch_type = "EC2"

  ordered_placement_strategy {
    type  = "binpack"
    field = "cpu"
  }

  lifecycle {
    ignore_changes = [task_definition]
  }

  placement_constraints {
    type       = "memberOf"
    expression = "ec2InstanceId==${aws_instance.ecs_instance.id}"
  }
}