from resnet50 import main_resnet50
from adversarial_attack import main_attack
from adversarial_defense import main_defense
from test_attack import main_attack_test


if __name__ == '__main__':
    main_resnet50()  #计算原始准确率
    #main_attack()  #生成对抗样本
    main_attack_test()  #计算对抗样本攻击后的准确率
    main_defense()  #计算对抗样本防御后的准确率