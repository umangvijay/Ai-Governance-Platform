"""
Dashboard Visualizer - Create charts and visualizations
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


class DashboardVisualizer:
    """Create visualizations for dashboard"""
    
    def __init__(self, output_dir: str = 'dashboard/outputs'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generated_charts = []
        
    def plot_model_performance(self, training_results: dict) -> str:
        """Plot model performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('ML Model Performance Metrics', fontsize=16, fontweight='bold')
        
        # Extract metrics
        models = []
        accuracies = []
        r2_scores = []
        
        for model_name, metrics in training_results.items():
            if metrics.get('status') == 'success':
                models.append(model_name.capitalize())
                
                if 'accuracy' in metrics:
                    accuracies.append(metrics['accuracy'])
                elif 'risk_r2' in metrics:
                    accuracies.append(metrics['risk_r2'])
                elif 'r2' in metrics:
                    accuracies.append(metrics['r2'])
                else:
                    accuracies.append(0)
        
        # Plot 1: Model Accuracy
        if models and accuracies:
            axes[0, 0].bar(models, accuracies, color='steelblue', alpha=0.7)
            axes[0, 0].set_title('Model Accuracy / R² Scores', fontweight='bold')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            for i, v in enumerate(accuracies):
                axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        # Plot 2: Training Status
        status_counts = {'Success': 0, 'Failed': 0, 'Skipped': 0}
        for metrics in training_results.values():
            status = metrics.get('status', 'unknown')
            if status in status_counts:
                status_counts[status.capitalize()] += 1
        
        colors = ['#2ecc71', '#e74c3c', '#f39c12']
        axes[0, 1].pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.0f%%',
                      colors=colors, startangle=90)
        axes[0, 1].set_title('Training Status Distribution', fontweight='bold')
        
        # Plot 3: Feature counts
        feature_counts = []
        feature_models = []
        for model_name, metrics in training_results.items():
            if 'features_used' in metrics:
                feature_models.append(model_name.capitalize())
                feature_counts.append(metrics['features_used'])
        
        if feature_models:
            axes[1, 0].barh(feature_models, feature_counts, color='coral', alpha=0.7)
            axes[1, 0].set_title('Features Used Per Model', fontweight='bold')
            axes[1, 0].set_xlabel('Number of Features')
        else:
            axes[1, 0].text(0.5, 0.5, 'No feature data', ha='center', va='center')
            axes[1, 0].set_title('Features Used Per Model', fontweight='bold')
        
        # Plot 4: Summary text
        axes[1, 1].axis('off')
        summary_text = "Model Training Summary\n\n"
        summary_text += f"Total Models: {len(training_results)}\n"
        summary_text += f"Successful: {status_counts['Success']}\n"
        summary_text += f"Failed: {status_counts['Failed']}\n"
        summary_text += f"Skipped: {status_counts['Skipped']}\n\n"
        summary_text += f"Average Accuracy: {np.mean(accuracies):.3f}" if accuracies else "No accuracy data"
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'model_performance.png'
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        self.generated_charts.append(str(output_path))
        logger.info(f"✓ Saved model performance chart: {output_path}")
        
        return str(output_path)
    
    def plot_predictions(self, predictions: dict) -> str:
        """Plot prediction results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Prediction Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Health risk forecast
        if 'health_forecast' in predictions:
            forecast = predictions['health_forecast']
            days = list(range(1, len(forecast) + 1))
            axes[0, 0].plot(days, forecast, marker='o', linewidth=2, color='red')
            axes[0, 0].fill_between(days, forecast, alpha=0.3, color='red')
            axes[0, 0].set_title('Health Risk Forecast (7 Days)', fontweight='bold')
            axes[0, 0].set_xlabel('Day')
            axes[0, 0].set_ylabel('Risk Score')
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'No health forecast', ha='center', va='center')
            axes[0, 0].set_title('Health Risk Forecast', fontweight='bold')
        
        # Plot 2: Infrastructure risk
        if 'infrastructure_risk' in predictions:
            risk_data = predictions['infrastructure_risk']
            locations = [f"Loc-{i+1}" for i in range(len(risk_data[:10]))]
            risks = risk_data[:10]
            
            colors_risk = ['red' if r > 0.7 else 'orange' if r > 0.4 else 'green' for r in risks]
            axes[0, 1].barh(locations, risks, color=colors_risk, alpha=0.7)
            axes[0, 1].set_title('Infrastructure Failure Risk (Top 10)', fontweight='bold')
            axes[0, 1].set_xlabel('Risk Probability')
            axes[0, 1].set_xlim(0, 1)
        else:
            axes[0, 1].text(0.5, 0.5, 'No infrastructure risk', ha='center', va='center')
            axes[0, 1].set_title('Infrastructure Failure Risk', fontweight='bold')
        
        # Plot 3: Demand forecast
        if 'demand_forecast' in predictions:
            forecast = predictions['demand_forecast']
            hours = [f['hour'] for f in forecast[:24]]
            demands = [f['predicted_demand'] for f in forecast[:24]]
            
            axes[1, 0].plot(hours, demands, marker='s', linewidth=2, color='blue')
            axes[1, 0].fill_between(hours, demands, alpha=0.3, color='blue')
            axes[1, 0].set_title('Service Demand Forecast (24 Hours)', fontweight='bold')
            axes[1, 0].set_xlabel('Hour')
            axes[1, 0].set_ylabel('Demand')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No demand forecast', ha='center', va='center')
            axes[1, 0].set_title('Service Demand Forecast', fontweight='bold')
        
        # Plot 4: Sentiment distribution
        if 'sentiment_distribution' in predictions:
            sentiment_data = predictions['sentiment_distribution']
            labels = list(sentiment_data.keys())
            sizes = [sentiment_data[k]['percentage'] for k in labels]
            colors_sent = ['#e74c3c', '#f39c12', '#2ecc71']
            
            axes[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_sent, startangle=90)
            axes[1, 1].set_title('Citizen Sentiment Distribution', fontweight='bold')
        else:
            axes[1, 1].text(0.5, 0.5, 'No sentiment data', ha='center', va='center')
            axes[1, 1].set_title('Citizen Sentiment Distribution', fontweight='bold')
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'predictions.png'
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        self.generated_charts.append(str(output_path))
        logger.info(f"✓ Saved predictions chart: {output_path}")
        
        return str(output_path)
    
    def plot_data_overview(self, data_stats: dict) -> str:
        """Plot data overview"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Dataset Overview', fontsize=16, fontweight='bold')
        
        # Plot 1: Records per dataset
        datasets = list(data_stats['datasets'].keys())
        records = [data_stats['datasets'][d]['records'] for d in datasets]
        
        axes[0].barh(datasets, records, color='teal', alpha=0.7)
        axes[0].set_title('Records Per Dataset', fontweight='bold')
        axes[0].set_xlabel('Number of Records')
        
        for i, v in enumerate(records):
            axes[0].text(v, i, f' {v:,}', va='center')
        
        # Plot 2: Summary pie chart
        total_records = data_stats['total_records']
        percentages = [(data_stats['datasets'][d]['records'] / total_records * 100) for d in datasets]
        
        axes[1].pie(percentages, labels=datasets, autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Dataset Distribution', fontweight='bold')
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'data_overview.png'
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        self.generated_charts.append(str(output_path))
        logger.info(f"✓ Saved data overview chart: {output_path}")
        
        return str(output_path)
    
    def generate_html_dashboard(self, title: str = "AI Governance Platform Dashboard") -> str:
        """Generate HTML dashboard"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }}
        .timestamp {{
            text-align: center;
            color: #7f8c8d;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
"""
        
        for chart_path in self.generated_charts:
            chart_name = Path(chart_path).stem.replace('_', ' ').title()
            html += f"""
    <div class="chart-container">
        <h2>{chart_name}</h2>
        <img src="{Path(chart_path).name}" alt="{chart_name}">
    </div>
"""
        
        html += f"""
    <div class="timestamp">
        Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
</body>
</html>
"""
        
        output_path = self.output_dir / 'dashboard.html'
        with open(output_path, 'w') as f:
            f.write(html)
        
        logger.info(f"✓ Generated HTML dashboard: {output_path}")
        
        return str(output_path)


if __name__ == "__main__":
    print("=" * 80)
    print("DASHBOARD VISUALIZER TEST")
    print("=" * 80)
    
    # Create visualizer
    viz = DashboardVisualizer()
    
    # Sample data
    training_results = {
        'health': {'status': 'success', 'risk_r2': 0.87, 'features_used': 7},
        'infrastructure': {'status': 'success', 'accuracy': 0.84, 'features_used': 7},
        'demand': {'status': 'success', 'r2': 0.82, 'features_used': 7},
        'sentiment': {'status': 'success', 'accuracy': 0.89, 'features_used': 2}
    }
    
    predictions = {
        'health_forecast': [50, 52, 48, 55, 53, 51, 54],
        'infrastructure_risk': [0.8, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.85, 0.70, 0.60],
        'demand_forecast': [{'hour': i, 'predicted_demand': 50 + i*2} for i in range(24)],
        'sentiment_distribution': {
            'Positive': {'percentage': 40.0},
            'Neutral': {'percentage': 35.0},
            'Negative': {'percentage': 25.0}
        }
    }
    
    data_stats = {
        'total_datasets': 4,
        'total_records': 4300,
        'datasets': {
            'health': {'records': 500},
            'infrastructure': {'records': 1000},
            'service_requests': {'records': 2000},
            'citizen_feedback': {'records': 800}
        }
    }
    
    # Generate charts
    print("\nGenerating visualizations...")
    viz.plot_model_performance(training_results)
    viz.plot_predictions(predictions)
    viz.plot_data_overview(data_stats)
    
    # Generate HTML dashboard
    dashboard_path = viz.generate_html_dashboard()
    
    print("\n" + "=" * 80)
    print(f"DASHBOARD GENERATED: {dashboard_path}")
    print("=" * 80)
    print(f"\nGenerated {len(viz.generated_charts)} charts")
    print("\nOpen dashboard.html in a web browser to view results")
