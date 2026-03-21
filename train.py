"""
STGCN 模型训练脚本示例
"""
import torch
import numpy as np
from pathlib import Path


def prepare_dummy_data(num_samples=100, sequence_length=30, num_nodes=17):
    """
    准备虚拟训练数据（实际使用时请替换为真实数据集）

    Args:
        num_samples: 样本数量
        sequence_length: 序列长度
        num_nodes: 节点数量

    Returns:
        (x, y): 训练数据和标签
    """
    # 模拟不同动作的关键点序列
    x = []
    y = []

    actions = ['stand', 'sit', 'walk', 'run', 'fall', 'bend', 'raise_hands']

    for _ in range(num_samples):
        action_idx = np.random.randint(0, len(actions))

        # 生成模拟关键点序列
        sequence = np.random.randn(sequence_length, num_nodes, 2).astype(np.float32)

        # 添加一些动作特征
        if actions[action_idx] == 'walk':
            sequence[:, 11:17, 1] += np.sin(np.linspace(0, 10, sequence_length))[:, None]
        elif actions[action_idx] == 'run':
            sequence[:, 11:17, 1] += np.sin(np.linspace(0, 20, sequence_length))[:, None] * 2

        x.append(sequence)
        y.append(action_idx)

    return np.array(x), np.array(y)


def train_stgcn(model, train_loader, epochs=50, learning_rate=0.001):
    """
    训练 STGCN 模型

    Args:
        model: STGCN 模型
        train_loader: 训练数据加载器
        epochs: 训练轮数
        learning_rate: 学习率
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # (B, T, V, C) -> (B, C, T, V)
            batch_x = batch_x.permute(0, 3, 1, 2)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

    return model


def main():
    """主训练流程"""
    from src.action.recognizer import STGCNModel

    # 超参数
    num_classes = 8
    sequence_length = 30
    num_nodes = 17
    batch_size = 32
    epochs = 50

    # 准备数据
    print("准备训练数据...")
    X, y = prepare_dummy_data(num_samples=500)

    # 分割训练/验证集
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # 转换为 Tensor
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train).long()
    X_val = torch.from_numpy(X_val)
    y_val = torch.from_numpy(y_val).long()

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 创建模型
    print("初始化模型...")
    model = STGCNModel(
        in_channels=2,
        num_classes=num_classes,
        num_nodes=num_nodes,
        temporal_window=sequence_length
    )

    # 训练
    print("开始训练...")
    model = train_stgcn(model, train_loader, epochs=epochs)

    # 保存模型
    save_path = Path(__file__).parent / "models" / "stgcn_model.pth"
    save_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存: {save_path}")


if __name__ == '__main__':
    main()
