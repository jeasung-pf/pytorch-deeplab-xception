aws ec2 run-instances \
    --image-id ami-0db78afd3d150fc18 \
    --security-group-ids sg-09ae22d52000db8f8 \
    --count 1 \
    --instance-type t2.medium \
    --key-name pf-dev-seoul2 \
    --query "Instances[0].InstanceId"
