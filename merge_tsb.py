import os
from glob import glob
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator


def load_scalar_events(logdir):
    """
    加载某个目录下的 scalar 数据（如 reward, loss）
    """
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()

    scalars = {}
    for tag in ea.Tags().get("scalars", []):
        scalars[tag] = ea.Scalars(tag)
    return scalars


def shift_scalars(scalars_dict, step_offset):
    """
    将 scalar 的 step 做偏移（为了拼接）
    """
    shifted = {}
    for tag, events in scalars_dict.items():
        shifted[tag] = []
        for e in events:
            # 将 step 增加 offset，保持连续
            shifted[tag].append((e.wall_time, e.step + step_offset, e.value))
    return shifted


def write_merged_scalars(all_shifted_scalars, output_dir):
    """
    将合并后的 scalars 写入新的 TensorBoard 日志文件
    """
    writer = SummaryWriter(log_dir=output_dir)

    for tag, entries in all_shifted_scalars.items():
        for wall_time, step, value in entries:
            writer.add_scalar(tag, value, step)

    writer.close()
    print(f"✅ 已保存合并日志到: {output_dir}")


def merge_tensorboard_runs(input_dirs, output_dir):
    """
    主流程：从多个输入日志目录中读取并合并数据
    """
    step_offset = 0
    merged_scalars = {}

    for log_dir in input_dirs:
        print(f"📂 读取: {log_dir}")
        scalars = load_scalar_events(log_dir)
        shifted = shift_scalars(scalars, step_offset)

        # 合并所有 tag 的 scalar 数据
        for tag, values in shifted.items():
            if tag not in merged_scalars:
                merged_scalars[tag] = []
            merged_scalars[tag].extend(values)

        # 更新 step 偏移量（保证连续）
        if scalars:
            sample_tag = next(iter(scalars))  # 任意一个 tag
            steps = [e.step for e in scalars[sample_tag]]
            step_offset += max(steps) if steps else 0

    write_merged_scalars(merged_scalars, output_dir)


if __name__ == "__main__":
    # 自动收集多个阶段的日志文件夹
    base_dir = "./tsb/SofaGrasp"
    output_dir = os.path.join(base_dir, "merged_run")

    # 匹配所有 PPO_xxx 子目录
    input_dirs = sorted(glob(os.path.join(base_dir, "PPO_*")))

    print("🚀 开始合并 TensorBoard 日志...")
    merge_tensorboard_runs(input_dirs, output_dir)

    print("✅ 完成！你可以使用以下命令查看合并结果：")
    print(f"   tensorboard --logdir={output_dir}")
