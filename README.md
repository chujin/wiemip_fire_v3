## What fire_explicit_v3.py does

Turns ML monthly fire probabilities into a TEM-ready annual explicit-fire NetCDF: where/when each pixel burns (mask + day-of-year), on the climate grid, with TEM-safe dtypes.

### Inputs (from mode)
- ${mode}_monthly_prob_Native_combined_rescaled_v2.nc : pred_prob  
- climate_${mode}.nc : Y/X template for padding  

There are four modes: stable, UKESM1-0-LL, IPSL-CM6A-LR, GFDL-ESM4

### Outputs
- explicit-fire_{mode}_v3.nc : native ML grid  
- explicit-fire_{mode}_v3_new.nc : padded to climate grid (use this for dvm-dos-tem)  
- explicit-fire_stable_full_v3.nc : only if mod == "stable"  

### Workflow
<img width="785" height="1117" alt="image" src="https://github.com/user-attachments/assets/0adc1f57-3314-41bb-a514-32ba16eaf841" />

### How to run
1. Edit CONFIG: one mod = "..." line   
2. python fire_explicit_v3.py   
3. Feed dvm-dos-tem the *_v3_new.nc (or stable *_full_v3.nc)    

#### Notes:  
- GCMs already have ~151 years (no step 5).  
- Seed is 42.  
- Peak RAM ~a few GB (safe on ~16 GB login nodes).
