import numpy as np

def read_tracks_with_hits(filename):
    """
    读取CSV文件，其中每个事件由两行组成：
      第1行: 一条轨迹的所有hits，每个hit为(docа, xm, xr, yr, z)
      第2行: 对应的轨迹初态 (x, y, tx, ty, q/p) at z = 229 in the tilted sector frame

    返回:
      hits_list: list[np.ndarray], 每个元素形状 [num_hits, 5]
      states: np.ndarray, 形状 [N, 5]
    """
    hits_list = []
    states = []

    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip() != ""]

    if len(lines) % 2 != 0:
        raise ValueError("文件行数必须是偶数，每个事件两行。")

    for i in range(0, len(lines), 2):
        # 第一行：hits
        hit_values = [float(x) for x in lines[i].split(",")]
        if len(hit_values) % 5 != 0:
            raise ValueError(f"第{i+1}行 hits 数量不是5的倍数: {len(hit_values)}")
        hits = np.array(hit_values, dtype=np.float32).reshape(-1, 5)  # [num_hits, 5]
        hits_list.append(hits)

        # 第二行：track state
        state_values = [float(x) for x in lines[i+1].split(",")]
        if len(state_values) != 5:
            raise ValueError(f"第{i+2}行轨迹初态必须有5个数，但得到{len(state_values)}")
        states.append(state_values)

    states = np.array(states, dtype=np.float32)  # [N, 5]
    return hits_list, states


# 示例使用：
if __name__ == "__main__":
    hits, states = read_tracks_with_hits("sample.csv")
    print(f"共读取 {len(hits)} 条轨迹")
    print(f"第一条轨迹 hits 形状: {hits[0].shape}")  # (num_hits, 5)
    print(f"第一条轨迹初态: {states[0]}")
