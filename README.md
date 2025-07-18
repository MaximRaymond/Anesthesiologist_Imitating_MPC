# Anesthesiologist_Imitating_MPC

Git repository for the paper named "MPC-based Anesthesiologists Imitating Control of Propofol and Remifentanil during Anesthesia Maintenance"

Tag version of the paper [here](https://github.com/MaximRaymond/Anesthesiologist_Imitating_MPC/releases/tag/conference-result).
If you use this code for research please cite Maxim Raymond, Kaouther Moussa, Mirko Fiacchini, Jimmy Lauber. ***"MPC-based Anesthesiologists Imitating Control of Propofol and Remifentanil during Anesthesia Maintenance."*** 2025. ⟨hal-05116145⟩

This code use the [Python Anesthesia Simulator](https://github.com/BobAubouin/Python_Anesthesia_Simulator) to simulate anesthesia and the [TIVA drug control](https://github.com/BobAubouin/TIVA_Drug_Control) as a benchmark MPC.

**Abstract:**
This paper suggests a new formulation of a Model Predictive Control (MPC) strategy, allowing to design propofol and remifentanil infusion profiles in order to control the Bispectal Index (BIS). This new formulation, based on a range cost, allows to reduce the sensitivity of the control profiles with respect to the BIS measurement noise. Furthermore, it allows to better represent the anesthesiologists behavior in practice, who usually do not over-react to small changes in the measured health indicators. The paper assesses numerically this formulation, by comparing its performance, in an uncertain setting, to standard set-point tracking based MPC strategies.


## Installation

Using python 3.12

```
pip install git+https://github.com/BobAubouin/TIVA_Drug_Control.git
pip install -r requirements.txt
```

## Usage

The codes were made using python 3.12

## Reproduce results

Run either optimize_script/optimize_range.py or optimize_script/optimize_nominal.py to generate the data.

To draw the results use optimize_script/draw_test.py and modify the lines 143 to 144:
```
control_type = 'range'
study_name = 'paper_final_range_realistic'
```
in order to make it correspond to the data generated.

To change the disturbance profile, in optimize_script/generate_simulation.py in line 171 to 172

```
dist_type = 'step'
# dist_type = 'realistic'
```

Uncomment the desired profile and comment the other ones.

## Authors

Maxim Raymond, Kaouther Moussa, Mirko Fiacchini, Jimmy Lauber

## License

GPL-3.0