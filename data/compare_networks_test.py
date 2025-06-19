import wntr
import numpy as np

def load_and_simulate(inp_file):
  try:
      wn = wntr.network.WaterNetworkModel(inp_file)
      sim = wntr.sim.EpanetSimulator(wn)
      results = sim.run_sim()
      return wn, results
  except Exception as e:
      raise RuntimeError(f"Failed to load or simulate {inp_file}: {e}")

def compare_results(res_trial, res_ref, tolerance=1e-3):
  comparisons = {}

  # Pressure comparison
  pressure_trial = res_trial.node['pressure']
  pressure_ref = res_ref.node['pressure']
  pressure_diff = abs(pressure_trial - pressure_ref).max().max()
  comparisons['pressure'] = pressure_diff < tolerance
  print(pressure_trial)

  # Demand comparison
  demand_trial = res_trial.node['demand']
  demand_ref = res_ref.node['demand']
  demand_diff = abs(demand_trial - demand_ref).max().max()
  comparisons['demand'] = demand_diff < tolerance

  # Flow comparison
  flow_trial = res_trial.link['flowrate']
  flow_ref = res_ref.link['flowrate']
  flow_diff = abs(flow_trial - flow_ref).max().max()
  comparisons['flowrate'] = flow_diff < tolerance

  return comparisons

def run(trial_file, benchmark_file):
  print(f"Comparing trial file {trial_file} and benchmark file {benchmark_file}")
  try:
      wn_trial, res_trial = load_and_simulate(trial_file)
  except RuntimeError as e:
      print(str(e))
      return

  try:
      wn_ref, res_ref = load_and_simulate(benchmark_file)
  except RuntimeError as e:
      print(f"Reference network could not be loaded: {e}")
      return

  results = compare_results(res_trial, res_ref)

  passed_all = all(results.values())
  print("\n--- UNIT TEST RESULTS ---")
  for key, passed in results.items():
      print(f"{key} comparison: {'PASSED' if passed else 'FAILED'}")

  if passed_all:
      print("\n✅ All unit tests passed. The models are essentially the same.")
  else:
      print("\n❌ Some unit tests failed. Differences were detected.")

run('network_0.inp','benchmark.inp') #worked with 13GB RAM in Colab