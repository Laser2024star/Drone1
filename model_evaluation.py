import os
from ultralytics import YOLO
import matplotlib.pyplot as plt

def evaluate_model():
    """
    评估训练好的best.pt模型，计算TP、FP、FN，Precision、Recall，绘制P-R曲线，计算AP
    """
    # 加载模型
    model_path = os.path.join('models', 'best.pt')
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return

    model = YOLO(model_path)

    # 数据路径（从训练args.yaml中获取）
    data_path = 'D:/Visdrone2019/Task1/visdrone.yaml'  # 请根据实际情况修改

    if not os.path.exists(data_path):
        print(f"❌ 数据配置文件不存在: {data_path}")
        print("请确保VisDrone数据集的YAML配置文件路径正确")
        return

    print("🚀 开始模型评估...")

    # 运行验证，生成plots
    results = model.val(data=data_path, plots=True, save_json=True)

    # 打印整体指标
    print("\n📊 整体评估指标:")
    print(f"Precision: {results.box.p.mean():.4f}")
    print(f"Recall: {results.box.r.mean():.4f}")
    print(f"mAP@0.5: {results.box.ap50.mean():.4f}")
    print(f"mAP@0.5:0.95: {results.box.ap.mean():.4f}")

    # 打印每类AP
    print("\n📈 每类平均精度 (AP):")
    class_names = model.names  # 获取类别名称
    for i, ap in enumerate(results.box.ap):
        class_name = class_names.get(i, f'Class {i}')
        print(f"{class_name}: {ap:.4f}")

    # 计算TP, FP, FN（从results中提取）
    tp = results.box.tp.sum() if hasattr(results.box, 'tp') else 0
    fp = results.box.fp.sum() if hasattr(results.box, 'fp') else 0
    fn = results.box.fn.sum() if hasattr(results.box, 'fn') else 0

    print("\n🔍 检测统计:")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")

    # 计算Precision和Recall（手动验证）
    precision_manual = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_manual = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"\n✅ 手动计算Precision: {precision_manual:.4f}")
    print(f"✅ 手动计算Recall: {recall_manual:.4f}")

    # 可视化P-R曲线
    # YOLO已经生成了P_curve.png在runs/val/exp/目录下
    val_dir = 'runs/val/exp'
    pr_curve_path = os.path.join(val_dir, 'P_curve.png')

    if os.path.exists(pr_curve_path):
        print(f"\n📊 P-R曲线已生成: {pr_curve_path}")
        # 显示图像
        img = plt.imread(pr_curve_path)
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Precision-Recall Curve')
        plt.show()
    else:
        print("❌ P-R曲线图像未找到，请检查runs/val/exp/目录")

    # 绘制自定义P-R曲线（不同置信度阈值）
    print("\n🔄 计算不同置信度下的P-R曲线...")
    conf_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    precisions = []
    recalls = []

    for conf in conf_thresholds:
        res = model.val(data=data_path, conf=conf, plots=False, verbose=False)
        precisions.append(res.box.p.mean())
        recalls.append(res.box.r.mean())

    # 绘制P-R曲线
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker='o', linestyle='-', color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve at Different Confidence Thresholds')
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    # 添加阈值标签
    for i, conf in enumerate(conf_thresholds):
        plt.annotate(f'{conf}', (recalls[i], precisions[i]), textcoords="offset points", xytext=(5,5), ha='left')

    plt.show()

    # 计算AP（近似，使用梯形积分）
    from scipy.integrate import trapz
    ap_approx = trapz(precisions, recalls)
    print(f"\n🎯 近似平均精度 (AP): {ap_approx:.4f}")

    print("\n✅ 评估完成！")

if __name__ == "__main__":
    evaluate_model()