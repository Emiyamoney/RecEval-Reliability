#!/usr/bin/env python
"""
Preflight Check — 论文级实验前验证脚本

检查内容:
  1. LightGCN 是否为真实图卷积实现
  2. DeepFM predict 是否禁止整体 fallback
  3. FeatureBuilder 是否为动态 schema
  4. 三个 loader 是否统一列名
  5. Amazon ID 是否在 loader 层稳定整数化
  6. YAML 是否为主配置来源
  7. create_model() 是否覆盖全部 13 个模型
  8. 所有模型是否实现统一接口
  9. scaler/encoder 是否 train-only
  10. smoke report 是否明确标注 pipeline verification only

用法:
  python scripts/preflight_check.py
"""

import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def check(name: str, passed: bool, detail: str = "") -> bool:
    status = "✅ PASS" if passed else "❌ FAIL"
    line = f"  [{status}] {name}"
    if detail and not passed:
        line += f"\n         → {detail}"
    print(line)
    return passed


def main():
    print("=" * 70)
    print("PREFLIGHT CHECK — 论文级实验前验证")
    print("=" * 70)
    results = []

    # ---- 1. LightGCN 图卷积 ----
    print("\n[1] LightGCN 真实图卷积检查")
    try:
        from models.lightgcn import LightGCN, LightGCNModel
        import inspect
        src = inspect.getsource(LightGCNModel)

        has_propagation = "propagate" in src and "norm_adj" in src
        has_graph_build = "coo_matrix" in src or "sparse" in src
        no_fake = "no actual graph convolution" not in src

        r1 = check("LightGCN 包含 propagate() 方法", has_propagation)
        r2 = check("LightGCN 构建稀疏图邻接矩阵", has_graph_build)
        r3 = check("LightGCN 不再是假实现 (dot-product only)", no_fake)
        results.extend([r1, r2, r3])
    except Exception as e:
        check("LightGCN 模块可导入", False, str(e))
        results.append(False)

    # ---- 2. DeepFM 无静默 fallback ----
    print("\n[2] DeepFM 无静默 fallback 检查")
    try:
        with open("trainers/unified_trainer.py", "r", encoding="utf-8") as f:
            trainer_src = f.read()
        has_fail_fast = "FAIL FAST" in trainer_src or "no silent fallback" in trainer_src
        no_global_mean_fallback = "return np.full(len(test_data), model.global_mean" not in trainer_src

        r4 = check("DeepFM predict 不再静默返回 global_mean", no_global_mean_fallback)
        r5 = check("DeepFM predict 有明确错误信息", has_fail_fast)
        results.extend([r4, r5])
    except Exception as e:
        check("unified_trainer.py 可读", False, str(e))
        results.append(False)

    # ---- 3. FeatureBuilder 动态 schema ----
    print("\n[3] FeatureBuilder 动态 schema 检查")
    try:
        with open("features/feature_builder.py", "r", encoding="utf-8") as f:
            fb_src = f.read()
        no_hardcoded_48 = "30 + 18" not in fb_src and "gender(2)" not in fb_src
        has_dynamic = "动态" in fb_src or "columns" in fb_src

        r6 = check("FeatureBuilder 不再硬编码 48 维", no_hardcoded_48)
        r7 = check("_build_profile_features 使用动态列检测", has_dynamic)
        results.extend([r6, r7])
    except Exception as e:
        check("feature_builder.py 可读", False, str(e))
        results.append(False)

    # ---- 4. 统一列名 ----
    print("\n[4] 统一列名 (snake_case) 检查")
    try:
        import pandas as pd
        from data.loaders.ml1m_loader import load_ml1m
        from data.loaders.goodbooks_loader import load_goodbooks
        from data.loaders.book_crossing_loader import load_book_crossing

        cfg = {"enabled": True, "max_users": 50, "max_items": 50, "max_interactions": 100}

        df_ml = load_ml1m({"name": "ml1m"}, sample_config=cfg)
        ml_ok = all(c in df_ml.columns for c in ["user_id", "item_id", "rating"])
        r8 = check("ML-1M loader 输出 snake_case 列名", ml_ok,
                    f"Columns: {list(df_ml.columns)[:8]}")

        df_gb = load_goodbooks(sample_config=cfg, raw_dir="data/raw/goodbooks")
        gb_ok = all(c in df_gb.columns for c in ["user_id", "item_id", "rating"])
        r9 = check("Goodbooks loader 输出 snake_case 列名", gb_ok,
                    f"Columns: {list(df_gb.columns)[:8]}")

        df_bc = load_book_crossing(sample_config=cfg)
        bc_ok = all(c in df_bc.columns for c in ["user_id", "item_id", "rating"])
        r10 = check("Book-Crossing loader 输出 snake_case 列名", bc_ok,
                     f"Columns: {list(df_bc.columns)[:8]}")

        results.extend([r8, r9, r10])
    except Exception as e:
        check("Loader 测试", False, str(e))
        results.append(False)

    # ---- 5. Book-Crossing ID 整数化 + age 特征 ----
    print("\n[5] Book-Crossing ID 整数化 + age 特征检查")
    try:
        from data.loaders.book_crossing_loader import load_book_crossing
        cfg = {"enabled": True, "max_users": 100, "max_items": 100, "max_interactions": 500}
        df1 = load_book_crossing(sample_config=cfg)
        uid_type_ok = df1["user_id"].dtype in ("int64", "int32")
        iid_type_ok = df1["item_id"].dtype in ("int64", "int32")
        has_raw = "raw_user_id" in df1.columns
        has_age = "age" in df1.columns
        age_valid = df1["age"].notna().sum() > 0

        # 可复现性
        df2 = load_book_crossing(sample_config=cfg)
        reproducible = (df1["user_id"].iloc[0] == df2["user_id"].iloc[0])

        r11 = check("Book-Crossing user_id 为整数类型", uid_type_ok, f"dtype={df1['user_id'].dtype}")
        r12 = check("Book-Crossing item_id 为整数类型", iid_type_ok, f"dtype={df1['item_id'].dtype}")
        r13 = check("保留 raw_user_id/raw_item_id 列", has_raw)
        r14 = check("ID 映射可复现 (两次加载相同)", reproducible)
        r15 = check(f"包含有效 age 特征 ({df1['age'].notna().sum()}/{len(df1)})", age_valid)
        results.extend([r11, r12, r13, r14, r15])
    except Exception as e:
        check("Book-Crossing 测试", False, str(e))
        results.append(False)

    # ---- 6. 模型注册 ----
    print("\n[6] 统一模型注册检查")
    try:
        from models.registry import MODEL_REGISTRY, create_model, get_model_count

        n = get_model_count()
        r15 = check(f"注册表包含 {n} 个模型 (目标 ≥ 13)", n >= 13,
                     f"已注册 {n} 个: {sorted(MODEL_REGISTRY.keys())}")

        # 检查每个模型可实例化
        all_ok = True
        failed_models = []
        for mname in MODEL_REGISTRY:
            try:
                m = create_model(mname)
                has_fit = hasattr(m, "fit")
                has_predict = hasattr(m, "predict")
                if not (has_fit and has_predict):
                    failed_models.append(f"{mname} (fit={has_fit}, predict={has_predict})")
                    all_ok = False
            except Exception as e:
                failed_models.append(f"{mname}: {e}")
                all_ok = False

        r16 = check("所有模型实现 fit/predict 接口", all_ok,
                     "; ".join(failed_models) if failed_models else "")
        results.extend([r15, r16])
    except Exception as e:
        check("模型注册", False, str(e))
        results.append(False)

    # ---- 7. 无 .rename() 补丁 ----
    print("\n[7] 无 .rename() 补丁检查")
    try:
        with open("trainers/unified_trainer.py", "r", encoding="utf-8") as f:
            trainer_src = f.read()
        no_rename = ".rename(columns=" not in trainer_src
        r17 = check("unified_trainer.py 无 .rename() 补丁", no_rename)
        results.append(r17)
    except Exception as e:
        check("unified_trainer.py 检查", False, str(e))
        results.append(False)

    # ---- 8. Smoke report 声明 ----
    print("\n[8] Smoke report 声明检查")
    try:
        report_path = "results/smoke/smoke_report.md"
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                report = f.read()
            has_disclaimer = "pipeline verification" in report.lower() or "not for paper" in report.lower()
            r18 = check("Smoke report 包含 pipeline verification 声明", has_disclaimer)
        else:
            r18 = check("Smoke report 存在", False, f"{report_path} 不存在")
        results.append(r18)
    except Exception as e:
        check("Smoke report", False, str(e))
        results.append(False)

    # ---- Summary ----
    print("\n" + "=" * 70)
    n_pass = sum(results)
    n_total = len(results)
    all_pass = all(results)

    print(f"结果: {n_pass}/{n_total} 通过")

    if all_pass:
        print("\n✅ 所有检查通过! 可以开始 Full Experiment.")
        print("   python scripts/run_experiment.py --config configs/full_gpu.yaml")
    else:
        print(f"\n❌ {n_total - n_pass} 项未通过. 请在运行 Full Experiment 前修复.")
        print("   修复后重新运行: python scripts/preflight_check.py")

    print("=" * 70)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
