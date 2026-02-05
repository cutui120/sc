
"""
代谢工程机器学习管道
用于预测优化麦角硫因产量的基因表达组合
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import r2_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

# 可选依赖
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False




    
    def advanced_feature_engineering(self, df_expression):
        """高级特征工程"""
        print("执行高级特征工程...")
        
        gene_cols = [col for col in df_expression.columns if col.startswith('Gene_')]
        X = df_expression[gene_cols].values
        y = df_expression['Yield'].values
        labeled_mask = ~np.isnan(y)
        
        df_features = pd.DataFrame()
        
        # 关键基因的表达比率
        for i in range(0, 10, 2):
            df_features[f'Ratio_{i}_{i+1}'] = X[:, i] / (X[:, i+1] + 1e-10)
        
        # 统计量
        df_features['Gene_Mean'] = np.mean(X, axis=1)
        df_features['Gene_Std'] = np.std(X, axis=1)
        df_features['Gene_CV'] = df_features['Gene_Std'] / (df_features['Gene_Mean'] + 1e-10)
        
        # PCA特征
        pca = PCA(n_components=5, random_state=42)
        X_pca = pca.fit_transform(X)
        for i in range(5):
            df_features[f'PCA_{i+1}'] = X_pca[:, i]
        
        # 核PCA特征
        kpca = KernelPCA(n_components=3, kernel='rbf', gamma=0.1, random_state=42)
        X_kpca = kpca.fit_transform(X)
        for i in range(3):
            df_features[f'KernelPCA_{i+1}'] = X_kpca[:, i]
        
        # 基因簇特征
        gene_clusterer = KMeans(n_clusters=5, random_state=42, n_init=10)
        gene_clusters = gene_clusterer.fit_predict(X.T)
        for cluster_id in range(5):
            cluster_genes = np.where(gene_clusters == cluster_id)[0]
            if len(cluster_genes) > 0:
                df_features[f'Cluster_{cluster_id}_Mean'] = np.mean(X[:, cluster_genes], axis=1)
        
        # 交互特征
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = poly.fit_transform(X[:, :10])
        
        if labeled_mask.sum() > 10:
            mi_scores = mutual_info_regression(X_poly[labeled_mask], y[labeled_mask], random_state=42)
            top_poly_indices = np.argsort(mi_scores)[-5:]
            for i, idx in enumerate(top_poly_indices):
                df_features[f'Top_Poly_{i+1}'] = X_poly[:, idx]
        
        # 代谢通量特征
        for i in range(3):
            df_features[f'Flux_{i+1}'] = np.random.normal(0, 1, len(X)) + X[:, i] * 0.5
        
        # 动态特征
        df_features['Expression_Gradient'] = np.gradient(X[:, 0])
        
        # 异常检测特征
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        outlier_scores = lof.fit_predict(X)
        df_features['Outlier_Score'] = outlier_scores
        
        X_all = np.hstack([X, df_features.values])
        
        print(f"特征工程完成：原始特征 {X.shape[1]} -> 总特征 {X_all.shape[1]}")
        
        return X_all, y, labeled_mask, df_features.columns.tolist()

    def semi_supervised_learning(self, X, y, labeled_mask, params=None):
        """半监督学习算法"""
        print("执行半监督学习...")
        
        X_labeled = X[labeled_mask]
        y_labeled = y[labeled_mask]
        X_unlabeled = X[~labeled_mask]
        
        # 训练初始模型
        if params is None:
            params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 4}
        
        base_model = GradientBoostingRegressor(
            n_estimators=params.get('n_estimators', 100),
            learning_rate=params.get('learning_rate', 0.1),
            max_depth=params.get('max_depth', 4),
            random_state=42
        )
        base_model.fit(X_labeled, y_labeled)
        
        # 自训练
        print("  执行自训练...")
        y_pred_unlabeled = base_model.predict(X_unlabeled)
        
        # 计算置信度 - 使用多个子模型的预测方差
        # GradientBoostingRegressor的estimators_是一维数组
        if hasattr(base_model, 'estimators_') and len(base_model.estimators_) > 0:
            predictions = []
            # 使用staged_predict获取不同阶段的预测
            for i, pred in enumerate(base_model.staged_predict(X_unlabeled)):
                if i % 10 == 0:  # 每10个估计器取一次
                    predictions.append(pred)
            if len(predictions) > 1:
                predictions = np.array(predictions)
                confidence = 1.0 / (np.std(predictions, axis=0) + 1e-10)
            else:
                # 使用距离作为置信度
                knn = NearestNeighbors(n_neighbors=5)
                knn.fit(X_labeled)
                distances, _ = knn.kneighbors(X_unlabeled)
                confidence = 1.0 / (np.mean(distances, axis=1) + 1e-10)
        else:
            knn = NearestNeighbors(n_neighbors=5)
            knn.fit(X_labeled)
            distances, _ = knn.kneighbors(X_unlabeled)
            confidence = 1.0 / (np.mean(distances, axis=1) + 1e-10)
        
        # 选择高置信度样本
        confidence_threshold = np.percentile(confidence, 70)
        high_conf_mask = confidence >= confidence_threshold
        
        if high_conf_mask.sum() > 0:
            print(f"  选择 {high_conf_mask.sum()} 个高置信度无标签样本进行伪标记")
            
            X_train_extended = np.vstack([X_labeled, X_unlabeled[high_conf_mask]])
            y_train_extended = np.concatenate([y_labeled, y_pred_unlabeled[high_conf_mask]])
            
            model = GradientBoostingRegressor(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 4),
                random_state=42
            )
            model.fit(X_train_extended, y_train_extended)
            print(f"  自训练完成：训练集大小 {len(X_labeled)} -> {len(X_train_extended)}")
        else:
            model = base_model
            y_train_extended = y_labeled
            high_conf_mask = np.zeros(X_unlabeled.shape[0], dtype=bool)
            print("  没有足够高置信度的样本，使用初始模型")
        
        # 特征选择
        print("  执行特征选择...")
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_mask = importances > np.mean(importances) * 0.1
            
            if feature_mask.sum() > 10:
                self.feature_selector = feature_mask
                X_selected = X[:, feature_mask]
                print(f"  特征选择：{X.shape[1]} -> {X_selected.shape[1]} 个特征")
                
                # 重新训练
                X_labeled_selected = X_selected[labeled_mask]
                
                # 构建扩展训练集的索引
                unlabeled_indices = np.where(~labeled_mask)[0]
                extended_indices = np.concatenate([
                    np.where(labeled_mask)[0],
                    unlabeled_indices[high_conf_mask]
                ])
                
                if len(extended_indices) > len(X_labeled):
                    X_train_selected = X_selected[extended_indices]
                    model.fit(X_train_selected, y_train_extended)
                else:
                    model.fit(X_labeled_selected, y_labeled)
            else:
                X_selected = X
                self.feature_selector = None
        else:
            X_selected = X
            self.feature_selector = None
        
        self.model = model
        return model, X_selected
    
    def optimize_hyperparameters(self, X, y, labeled_mask, n_trials=30):
        """超参数优化"""
        print("优化超参数...")
        
        if not HAS_OPTUNA:
            print("  Optuna未安装，使用默认参数")
            return self.semi_supervised_learning(X, y, labeled_mask)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 8)
            }
            
            model, X_selected = self.semi_supervised_learning(X, y, labeled_mask, params)
            
            X_labeled = X_selected[labeled_mask]
            y_labeled = y[labeled_mask]
            
            kf = KFold(n_splits=min(5, labeled_mask.sum()), shuffle=True, random_state=42)
            scores = cross_val_score(model, X_labeled, y_labeled, cv=kf, scoring='r2')
            
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        print(f"最佳超参数：{study.best_params}")
        print(f"最佳R²分数：{study.best_value:.4f}")
        
        best_model, X_selected = self.semi_supervised_learning(X, y, labeled_mask, study.best_params)
        
        return best_model, X_selected

    def dbtl_cycle(self, df_expression, n_cycles=5, n_new_strains=10):
        """执行DBTL循环"""
        print("\n" + "="*60)
        print("开始DBTL循环")
        print("="*60)
        
        X, y, labeled_mask, feature_names = self.advanced_feature_engineering(df_expression)
        all_results = []
        
        for cycle in range(1, n_cycles + 1):
            print(f"\n{'='*30}")
            print(f"DBTL循环 #{cycle}")
            print(f"{'='*30}")
            
            print("\n1. 设计阶段：优化代谢途径")
            
            if cycle == 1:
                model, X_selected = self.optimize_hyperparameters(X, y, labeled_mask, n_trials=20)
            else:
                model, X_selected = self.semi_supervised_learning(X, y, labeled_mask)
            
            current_best_idx = np.nanargmax(y)
            current_best_yield = y[current_best_idx]
            print(f"当前最佳产量：{current_best_yield:.2f}")
            
            print("搜索最优基因表达组合...")
            optimal_expression = self.find_optimal_expression(model, X_selected, df_expression)
            
            print("\n2. 构建阶段：设计新菌株")
            new_strains = self.design_new_strains(optimal_expression, n_new_strains)
            
            print("\n3. 测试阶段：预测新菌株产量")
            predicted_yields, uncertainty = self.predict_new_strains(model, new_strains, X_selected)
            
            best_new_idx = np.argmax(predicted_yields)
            best_new_yield = predicted_yields[best_new_idx]
            best_new_strain = new_strains[best_new_idx]
            
            print(f"预测最佳新菌株产量：{best_new_yield:.2f}")
            print(f"不确定性：{uncertainty[best_new_idx]:.2f}")
            
            print("\n4. 学习阶段：整合新数据")
            actual_yield = best_new_yield * np.random.uniform(0.8, 1.0)
            
            new_row = df_expression.iloc[0].copy()
            gene_cols = [col for col in df_expression.columns if col.startswith('Gene_')]
            
            for i, col in enumerate(gene_cols[:len(best_new_strain)]):
                new_row[col] = best_new_strain[i]
            
            new_row['Yield'] = actual_yield
            new_row['Strain_ID'] = f'New_Strain_{cycle:02d}'
            new_row['Labeled'] = True
            new_row['Culture_Condition'] = 'Optimized'
            new_row['Growth_Phase'] = 'Log'
            new_row['OD600'] = np.random.uniform(1.5, 2.5)
            
            df_expression = pd.concat([df_expression, pd.DataFrame([new_row])], ignore_index=True)
            X, y, labeled_mask, feature_names = self.advanced_feature_engineering(df_expression)
            
            cycle_result = {
                'cycle': cycle,
                'current_best': current_best_yield,
                'predicted_best': best_new_yield,
                'actual_yield': actual_yield,
                'improvement': actual_yield - current_best_yield,
                'uncertainty': uncertainty[best_new_idx],
                'n_labeled': labeled_mask.sum()
            }
            all_results.append(cycle_result)
            
            print(f"实际产量：{actual_yield:.2f}")
            print(f"产量提升：{cycle_result['improvement']:.2f}")
            
            if actual_yield > current_best_yield * 1.05:
                self.optimal_expression = optimal_expression
                print("✓ 发现改进的表达组合！")
        
        print("\n" + "="*60)
        print("DBTL循环总结")
        print("="*60)
        
        results_df = pd.DataFrame(all_results)
        print(results_df.to_string())
        
        self.plot_dbtl_progress(results_df)
        
        return df_expression, results_df
    
    def find_optimal_expression(self, model, X_selected, df_expression):
        """寻找最佳基因表达组合"""
        gene_cols = [col for col in df_expression.columns if col.startswith('Gene_')]
        n_key_genes = 10
        
        current_data = df_expression[gene_cols[:n_key_genes]].values
        expression_min = np.min(current_data, axis=0)
        expression_max = np.max(current_data, axis=0)
        expression_mean = np.mean(current_data, axis=0)
        expression_std = np.std(current_data, axis=0)
        
        def objective_function(gene_expressions):
            sample = np.zeros(X_selected.shape[1])
            sample[:n_key_genes] = gene_expressions
            if len(sample) > n_key_genes:
                sample[n_key_genes:] = np.mean(X_selected, axis=0)[n_key_genes:]
            return model.predict([sample])[0]
        
        best_yield = -np.inf
        best_expression = expression_mean.copy()
        
        for _ in range(1000):
            candidate = np.random.normal(expression_mean, expression_std * 0.5)
            candidate = np.clip(candidate, expression_min * 0.8, expression_max * 1.2)
            yield_pred = objective_function(candidate)
            
            if yield_pred > best_yield:
                best_yield = yield_pred
                best_expression = candidate
        
        print(f"预测最优产量：{best_yield:.2f}")
        return best_expression
    
    def design_new_strains(self, optimal_expression, n_strains=10):
        """设计新菌株"""
        print(f"设计 {n_strains} 个新菌株...")
        
        new_strains = [optimal_expression]
        for i in range(n_strains - 1):
            noise = np.random.normal(0, 0.1 * (i + 1), len(optimal_expression))
            new_strains.append(optimal_expression + noise)
        
        return np.array(new_strains)
    
    def predict_new_strains(self, model, new_strains, X_selected):
        """预测新菌株产量"""
        predictions = []
        uncertainties = []
        
        for strain in new_strains:
            sample = np.zeros(X_selected.shape[1])
            sample[:len(strain)] = strain
            if len(sample) > len(strain):
                sample[len(strain):] = np.mean(X_selected, axis=0)[len(strain):]
            
            pred = model.predict([sample])[0]
            predictions.append(pred)
            
            # 使用staged_predict估计不确定性
            if hasattr(model, 'staged_predict'):
                staged_preds = list(model.staged_predict([sample]))
                if len(staged_preds) > 10:
                    recent_preds = staged_preds[-10:]
                    uncertainty = np.std([p[0] for p in recent_preds])
                else:
                    uncertainty = 0.5
            else:
                knn = NearestNeighbors(n_neighbors=5)
                knn.fit(X_selected)
                distances, _ = knn.kneighbors([sample])
                uncertainty = np.mean(distances)
            
            uncertainties.append(uncertainty)
        
        return np.array(predictions), np.array(uncertainties)

    def plot_dbtl_progress(self, results_df):
        """可视化DBTL循环进展"""
        if not HAS_MATPLOTLIB:
            print("警告：未安装matplotlib，跳过可视化")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 产量进展
        axes[0, 0].plot(results_df['cycle'], results_df['current_best'], 'b-o', label='当前最佳')
        axes[0, 0].plot(results_df['cycle'], results_df['predicted_best'], 'g--o', label='预测最佳')
        axes[0, 0].plot(results_df['cycle'], results_df['actual_yield'], 'r-s', label='实际产量')
        axes[0, 0].set_xlabel('DBTL循环')
        axes[0, 0].set_ylabel('麦角硫因产量')
        axes[0, 0].set_title('DBTL循环产量进展')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 提升情况
        colors = ['green' if x > 0 else 'red' for x in results_df['improvement']]
        axes[0, 1].bar(results_df['cycle'], results_df['improvement'], color=colors)
        axes[0, 1].set_xlabel('DBTL循环')
        axes[0, 1].set_ylabel('产量提升')
        axes[0, 1].set_title('每轮循环的产量提升')
        axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 不确定性
        axes[1, 0].plot(results_df['cycle'], results_df['uncertainty'], 'm-o')
        axes[1, 0].set_xlabel('DBTL循环')
        axes[1, 0].set_ylabel('预测不确定性')
        axes[1, 0].set_title('预测不确定性变化')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 有标签数据量
        axes[1, 1].plot(results_df['cycle'], results_df['n_labeled'], 'c-o')
        axes[1, 1].set_xlabel('DBTL循环')
        axes[1, 1].set_ylabel('有标签样本数')
        axes[1, 1].set_title('有标签数据积累')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('z2/dbtl_progress.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("图表已保存: z2/dbtl_progress.png")
    
    def interpret_results(self, df_expression, model):
        """解释模型结果"""
        print("\n" + "="*60)
        print("结果解释与关键基因识别")
        print("="*60)
        
        gene_cols = [col for col in df_expression.columns if col.startswith('Gene_')]
        n_key_genes = 10
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            key_gene_importance = importances[:n_key_genes]
            
            importance_df = pd.DataFrame({
                'Gene': gene_cols[:n_key_genes],
                'Importance': key_gene_importance
            }).sort_values('Importance', ascending=False)
            
            print("\n关键基因重要性排序：")
            print(importance_df.to_string())
        
        if self.optimal_expression is not None:
            print(f"\n推荐的最优基因表达组合（前{len(self.optimal_expression)}个关键基因）：")
            for i, value in enumerate(self.optimal_expression[:10]):
                gene_name = gene_cols[i] if i < len(gene_cols) else f"Key_Gene_{i}"
                print(f"  {gene_name}: {value:.3f}")
            
            mean_expression = np.mean(df_expression[gene_cols[:len(self.optimal_expression)]].values, axis=0)
            changes = (self.optimal_expression - mean_expression) / (np.abs(mean_expression) + 1e-10) * 100
            
            print("\n相对于平均表达的变化百分比：")
            for i, change in enumerate(changes[:10]):
                gene_name = gene_cols[i] if i < len(gene_cols) else f"Key_Gene_{i}"
                direction = "上调" if change > 0 else "下调"
                print(f"  {gene_name}: {direction} {abs(change):.1f}%")
    
    def run_full_pipeline(self):
        """运行完整管道"""
        print("启动麦角硫因产量优化机器学习管道")
        print("="*60)
        
        df_expression = self.generate_synthetic_data()
        df_expression_final, results = self.dbtl_cycle(df_expression, n_cycles=5)
        self.interpret_results(df_expression_final, self.model)
        
        print("\n" + "="*60)
        print("最终工程建议")
        print("="*60)
        
        if self.optimal_expression is not None:
            print("推荐进行以下基因工程改造：")
            
            gene_cols = [col for col in df_expression_final.columns if col.startswith('Gene_')]
            mean_expression = np.mean(df_expression_final[gene_cols[:len(self.optimal_expression)]].values, axis=0)
            
            for i in range(min(10, len(self.optimal_expression))):
                current = mean_expression[i]
                target = self.optimal_expression[i]
                change_pct = (target - current) / (np.abs(current) + 1e-10) * 100
                
                if abs(change_pct) > 10:
                    direction = "上调" if change_pct > 0 else "下调"
                    action = "过表达" if change_pct > 0 else "抑制表达"
                    
                    print(f"\n{gene_cols[i]}:")
                    print(f"  当前表达: {current:.2f}")
                    print(f"  目标表达: {target:.2f}")
                    print(f"  建议: {action} ({direction} {abs(change_pct):.1f}%)")
        
        print("\n" + "="*60)
        print("管道执行完成！")
        print("="*60)
        
        return df_expression_final, results, self.model


if __name__ == "__main__":
    pipeline = MetaboMLPipeline(n_genes=50, n_samples=200)
    df_final, results, final_model = pipeline.run_full_pipeline()
    
    # 保存结果到z2目录
    df_final.to_csv('z2/optimized_strains.csv', index=False, encoding='utf-8-sig')
    results.to_csv('z2/dbtl_results.csv', index=False, encoding='utf-8-sig')
    
    print("\n结果已保存到文件：")
    print("  - z2/optimized_strains.csv: 优化后的菌株数据")
    print("  - z2/dbtl_results.csv: DBTL循环结果")
    print("  - z2/dbtl_progress.png: DBTL进展图")

