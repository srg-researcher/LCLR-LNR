# A Lightweight and Robust Framework for Label Noise Mitigation in Centralized and Federated Learning

## Requirements
 - see requirements.txt

## Examples
 - Example run for LCLR-LNR (CIFAR100 with 20% symmetric noise)
    ```ruby
    python /LCLR-LNR/main.py --dataset cifar100 --num_class 100 --data_path ./data/cifar10 --noise_mode 'sym' --r 0.2 --window_length 5 --variance_window 30 --starting_percent 0.5 --increase 1.5 --step_length 40 --num_epochs 300
    ```
 - Example run for LCLR-LNR (CIFAR10 with 10% asymmetric noise)
    ```ruby
    python /LCLR-LNR/main.py --dataset cifar10 --num_class 10 --data_path ./data/cifar10 --noise_mode 'asym' --r 0.2 --window_length 5 --variance_window 30 --starting_percent 0.5 --increase 1.5 --step_length 40 --num_epochs 300
    ```
 - Example run for LCLR-LNR-FL (CIFAR10 with 20% symmetric noise, 100 users, active rate 0.01 )
    ```ruby
    python /LCLR-LNR-FL/main.py --data_name=CIFAR10 --model_name=resnet18 --subset_ratio=-1 --noisy_ratio=0.2 --variance_window=30 --noise_mode='sym' --control_name 1_100_0.1_iid_fix_a1_bn_1_1
   ```
 - Example run for LCLR-LNR-FL (CIFAR10 with 10% asymmetric noise, 50 users, active rate 0.01 )
   ```ruby
   python /LCLR-LNR-FL/main.py --data_name=CIFAR10 --model_name=resnet18 --subset_ratio=-1 --noisy_ratio=0.2 --variance_window=30 --noise_mode='asym' --control_name 1_50_0.1_iid_fix_a1_bn_1_1
   ```
 - Example run for LCLR-LNR-FL (CIFAR10 with 25% pairflip noise, 100 users, active rate 0.01 )
   ```ruby
   python /LCLR-LNR-FL/main.py --data_name=CIFAR10 --model_name=resnet18 --subset_ratio=-1 --noisy_ratio=0.25 --variance_window=30 --noise_mode='pairflip' --control_name 1_100_0.1_iid_fix_a1_bn_1_1
   ```


## Acknowledgement
- LCLR-LNR is built upon [UNICON](https://arxiv.org/pdf/2203.14542).
- LCLR-LNR-FL is built upon [HeteroFL](https://arxiv.org/abs/2010.01264).
