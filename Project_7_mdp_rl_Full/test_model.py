import numpy as np
import nn
import model

# 测试DeepQNetwork的基本功能
def test_deep_q_network():
    print("测试DeepQNetwork...")
    
    # 创建网络
    state_dim = 10
    action_dim = 4
    dqn = model.DeepQNetwork(state_dim, action_dim)
    
    print(f"状态维度: {dqn.state_size}")
    print(f"动作维度: {dqn.num_actions}")
    print(f"学习率: {dqn.learning_rate}")
    print(f"训练游戏数: {dqn.numTrainingGames}")
    print(f"批次大小: {dqn.batch_size}")
    print(f"参数数量: {len(dqn.parameters)}")
    
    # 测试前向传播
    batch_size = 5
    states = nn.Constant(np.random.randn(batch_size, state_dim).astype(np.float64))
    
    print("\n测试前向传播...")
    output = dqn.run(states)
    print(f"输入形状: {states.data.shape}")
    print(f"输出形状: {output.data.shape}")
    print(f"输出范围: [{output.data.min():.4f}, {output.data.max():.4f}]")
    
    # 测试损失计算
    print("\n测试损失计算...")
    Q_target = nn.Constant(np.random.randn(batch_size, action_dim).astype(np.float64))
    loss = dqn.get_loss(states, Q_target)
    print(f"损失值: {loss.data}")
    
    # 测试梯度更新
    print("\n测试梯度更新...")
    try:
        dqn.gradient_update(states, Q_target)
        print("梯度更新成功")
    except Exception as e:
        print(f"梯度更新失败: {e}")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_deep_q_network() 