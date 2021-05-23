# black_swan_strategy
Black Swan Strategy simulation to validate whether we should do it.

Usage:
============================================================
1. Install all requirements from requirements.txt
2. Add BLACK_SWAN_PATH to env vars and let it point to this directory.
2. Run the automated tests with python3 -m unittest discover tests
3. Run the Black-Scholes-tests with python3 black_scholes_model.py and assess validity.
4. Run the volatility-tests with python3 volatility.py adn assess validity.
5. Open a notebook and generate a simulation with sim = Simulation(path, SimulationOptions)
6. Run sim.run() to get the pay_outs.
7. Analyse the pay_outs per day or their payouts.cumsum(axis=0). Go nuts.


Note: All CSVs go into the prices-folder.
