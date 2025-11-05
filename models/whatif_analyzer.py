"""
What-If Scenario Analyzer - Test different scenarios and forecast outcomes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhatIfAnalyzer:
    """Analyze what-if scenarios for governance decisions"""
    
    def __init__(self):
        self.baseline_data = None
        self.scenarios = []
        
    def set_baseline(self, data: Dict[str, Any]):
        """Set baseline scenario"""
        self.baseline_data = data
        logger.info("Baseline scenario set")
    
    def create_scenario(self, 
                       name: str,
                       changes: Dict[str, float],
                       forecast_days: int = 30) -> Dict[str, Any]:
        """
        Create a what-if scenario
        
        Args:
            name: Scenario name
            changes: Dictionary of parameter changes (e.g., {'beds': 1.2, 'doctors': 1.5})
            forecast_days: Number of days to forecast (1-30)
        
        Returns:
            Scenario results with predictions
        """
        if not self.baseline_data:
            logger.error("No baseline data set")
            return None
        
        scenario = {
            'name': name,
            'changes': changes,
            'forecast_days': forecast_days,
            'results': {}
        }
        
        # Apply changes to baseline
        modified_data = self.baseline_data.copy()
        for param, multiplier in changes.items():
            if param in modified_data:
                modified_data[param] *= multiplier
        
        # Generate forecast
        scenario['results']['modified_data'] = modified_data
        scenario['results']['forecast'] = self._generate_forecast(modified_data, forecast_days)
        scenario['results']['impact'] = self._calculate_impact(modified_data, forecast_days)
        
        self.scenarios.append(scenario)
        
        logger.info(f"Created scenario: {name}")
        return scenario
    
    def _generate_forecast(self, data: Dict[str, Any], days: int) -> List[float]:
        """Generate forecast based on modified parameters"""
        base_value = data.get('base_demand', 100)
        growth_rate = data.get('growth_rate', 0.02)
        
        forecast = []
        for day in range(1, days + 1):
            # Simple exponential growth with noise
            value = base_value * (1 + growth_rate) ** day
            noise = np.random.normal(0, value * 0.1)
            forecast.append(max(0, value + noise))
        
        return forecast
    
    def _calculate_impact(self, data: Dict[str, Any], days: int) -> Dict[str, Any]:
        """Calculate impact metrics"""
        forecast = self._generate_forecast(data, days)
        baseline_forecast = self._generate_forecast(self.baseline_data, days)
        
        return {
            'average_change': np.mean(forecast) - np.mean(baseline_forecast),
            'peak_change': max(forecast) - max(baseline_forecast),
            'total_change': sum(forecast) - sum(baseline_forecast),
            'percent_change': ((np.mean(forecast) - np.mean(baseline_forecast)) / 
                             np.mean(baseline_forecast) * 100)
        }
    
    def compare_scenarios(self) -> pd.DataFrame:
        """Compare all scenarios"""
        if not self.scenarios:
            logger.warning("No scenarios to compare")
            return pd.DataFrame()
        
        comparison = []
        for scenario in self.scenarios:
            comparison.append({
                'Scenario': scenario['name'],
                'Forecast Days': scenario['forecast_days'],
                'Avg Impact': scenario['results']['impact']['average_change'],
                'Peak Impact': scenario['results']['impact']['peak_change'],
                'Total Impact': scenario['results']['impact']['total_change'],
                'Percent Change': scenario['results']['impact']['percent_change']
            })
        
        return pd.DataFrame(comparison)
    
    def get_scenario(self, name: str) -> Dict[str, Any]:
        """Get specific scenario by name"""
        for scenario in self.scenarios:
            if scenario['name'] == name:
                return scenario
        return None
    
    def clear_scenarios(self):
        """Clear all scenarios"""
        self.scenarios = []
        logger.info("All scenarios cleared")


class HealthWhatIfAnalyzer(WhatIfAnalyzer):
    """What-if analyzer for health scenarios"""
    
    def analyze_bed_increase(self, percent_increase: float, days: int = 30) -> Dict:
        """Analyze impact of increasing hospital beds"""
        return self.create_scenario(
            name=f"Increase Beds by {percent_increase}%",
            changes={'beds': 1 + (percent_increase / 100)},
            forecast_days=days
        )
    
    def analyze_staff_increase(self, 
                              doctors_percent: float,
                              nurses_percent: float,
                              days: int = 30) -> Dict:
        """Analyze impact of increasing healthcare staff"""
        return self.create_scenario(
            name=f"Increase Doctors {doctors_percent}%, Nurses {nurses_percent}%",
            changes={
                'doctors': 1 + (doctors_percent / 100),
                'nurses': 1 + (nurses_percent / 100)
            },
            forecast_days=days
        )
    
    def analyze_outbreak_response(self, severity: str, days: int = 30) -> Dict:
        """Analyze outbreak response scenarios"""
        severity_levels = {
            'mild': {'patients': 1.2, 'emergency_cases': 1.1},
            'moderate': {'patients': 1.5, 'emergency_cases': 1.3},
            'severe': {'patients': 2.0, 'emergency_cases': 1.8}
        }
        
        changes = severity_levels.get(severity, severity_levels['moderate'])
        
        return self.create_scenario(
            name=f"Outbreak Response - {severity.capitalize()}",
            changes=changes,
            forecast_days=days
        )


class InfrastructureWhatIfAnalyzer(WhatIfAnalyzer):
    """What-if analyzer for infrastructure scenarios"""
    
    def analyze_response_time_improvement(self, percent_improvement: float, days: int = 30) -> Dict:
        """Analyze impact of improving response times"""
        return self.create_scenario(
            name=f"Improve Response Time by {percent_improvement}%",
            changes={'response_time': 1 - (percent_improvement / 100)},
            forecast_days=days
        )
    
    def analyze_maintenance_increase(self, percent_increase: float, days: int = 30) -> Dict:
        """Analyze impact of increasing maintenance resources"""
        return self.create_scenario(
            name=f"Increase Maintenance by {percent_increase}%",
            changes={'maintenance_budget': 1 + (percent_increase / 100)},
            forecast_days=days
        )


class DemandWhatIfAnalyzer(WhatIfAnalyzer):
    """What-if analyzer for demand forecasting scenarios"""
    
    def analyze_service_expansion(self, service_type: str, percent_increase: float, days: int = 30) -> Dict:
        """Analyze impact of expanding services"""
        return self.create_scenario(
            name=f"Expand {service_type} by {percent_increase}%",
            changes={'service_capacity': 1 + (percent_increase / 100)},
            forecast_days=days
        )
    
    def analyze_seasonal_surge(self, surge_factor: float, days: int = 30) -> Dict:
        """Analyze seasonal demand surge"""
        return self.create_scenario(
            name=f"Seasonal Surge (Factor: {surge_factor}x)",
            changes={'base_demand': surge_factor},
            forecast_days=days
        )


if __name__ == "__main__":
    print("=" * 80)
    print("WHAT-IF SCENARIO ANALYZER TEST")
    print("=" * 80)
    
    # Test Health What-If Analysis
    print("\n### HEALTH SCENARIOS ###")
    health_analyzer = HealthWhatIfAnalyzer()
    
    # Set baseline
    health_analyzer.set_baseline({
        'beds': 1000,
        'doctors': 100,
        'nurses': 200,
        'patients': 800,
        'base_demand': 100,
        'growth_rate': 0.02
    })
    
    # Create scenarios
    scenario1 = health_analyzer.analyze_bed_increase(20, days=30)
    scenario2 = health_analyzer.analyze_staff_increase(15, 25, days=30)
    scenario3 = health_analyzer.analyze_outbreak_response('moderate', days=30)
    
    # Compare scenarios
    comparison = health_analyzer.compare_scenarios()
    print("\nScenario Comparison:")
    print(comparison.to_string(index=False))
    
    # Show detailed forecast for one scenario
    print(f"\n\nDetailed Forecast for: {scenario1['name']}")
    forecast = scenario1['results']['forecast']
    print(f"Days 1-7: {[f'{v:.0f}' for v in forecast[:7]]}")
    print(f"Days 8-14: {[f'{v:.0f}' for v in forecast[7:14]]}")
    print(f"Days 15-21: {[f'{v:.0f}' for v in forecast[14:21]]}")
    print(f"Days 22-30: {[f'{v:.0f}' for v in forecast[21:30]]}")
    
    impact = scenario1['results']['impact']
    print(f"\nImpact Analysis:")
    print(f"  Average Change: {impact['average_change']:.2f}")
    print(f"  Peak Change: {impact['peak_change']:.2f}")
    print(f"  Percent Change: {impact['percent_change']:.2f}%")
    
    print("\n" + "=" * 80)
    print("WHAT-IF ANALYZER READY")
    print("=" * 80)
