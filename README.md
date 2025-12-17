# Infrasound Data Analysis - Tatun Volcano Group

## Project Overview

This project analyzes high-density infrasound data collected from the Tatun Volcano Group (大屯火山群) in Taiwan. The analysis focuses on signal processing, feature extraction, and pattern recognition to identify and characterize volcanic infrasound signatures.

**Data Period**: March 3-19, 2023 (17 days)  
**Monitoring Sites**: Two main stations - SYK (6 sensors) and YD (5 sensors)  
**Sampling Rate**: 100 Hz

> **Note**: Raw data and result images are not included in this repository due to data privacy restrictions and large file sizes. Please refer to the project poster (Chinese) ([大屯火山群高密度次聲波資料分析_林奕丞_期末海報.pdf](報告/大屯火山群高密度次聲波資料分析_林奕丞_期末海報.pdf)) for visualizations and analysis results.

## Analysis Pipeline

The analysis workflow consists of band power and K-Means/DBSCAN implementations (my primary work):

### Stage 1: Band Power Analysis (Files 6_*.py)

#### [6_band_power.py](6_band_power.py)
Basic band power calculation for three frequency bands:
- 0.1-1 Hz (microbaroms, meteorological signals)
- 1-10 Hz (volcanic tremor, explosion signals)
- 10-20 Hz (high-frequency volcanic activity)

**Features**:
- Computes Power Spectral Density (PSD) from FFT data
- Applies z-score normalization
- Generates time series plots for each band

**Usage**:
```bash
python 6_band_power.py <th_dir> <fft_dir> <output_dir> <time_length>
```

#### [6b_band_power.py](6b_band_power.py)
Extended band power analysis with configurable frequency bands:
- Flexible band configuration
- Multi-station processing
- Peak detection using `scipy.signal.argrelextrema`

#### [6c_band_power.py](6c_band_power.py)
Advanced band power analysis with comprehensive visualization:
- Processes all sensors for SYK (01-06) and YD (04, 05, 06, 07, 21)
- Handles missing data with NaN interpolation
- Computes band power across entire dataset
- Peak/valley detection for temporal pattern identification
- Exports results in both CSV and NPY formats

**Key Functions**:
```python
def band_power(th, fft, band_start, band_end):
    """Calculate total power in specified frequency band
    
    Args:
        th: Time series data
        fft: FFT coefficients
        band_start: Lower frequency bound (Hz)
        band_end: Upper frequency bound (Hz)
    
    Returns:
        Total power in the specified band
    """
```

#### [6c_psd.py](6c_psd.py)
Power Spectral Density analysis using Welch's method:
- Uses `scipy.signal.welch` for robust PSD estimation
- Reduces noise through overlapping windowing
- Better frequency resolution for detailed spectral analysis

#### [6d_.py](6d_.py)
Variant implementation for specific band power computations with different parameter configurations.

### Stage 2: Clustering Analysis (Files 7_*.py)

#### [7_clustering.py](7_clustering.py)
K-means clustering for temporal pattern recognition:

**Features**:
- K-means++ initialization for better convergence
- Configurable number of clusters (default: 3)
- Cluster visualization with color-coded time spans
- Identifies distinct activity states

**Workflow**:
1. Load band power data from Stage 1
2. Impute missing values using mean strategy
3. Apply K-means clustering
4. Generate cluster labels and centers
5. Create visualizations:
   - Time series with cluster-colored background
   - 2D scatter plot of cluster distribution
   - Peak markers overlay

**Usage**:
```bash
python 7_clustering.py <place_name> <source_dir> <output_dir> <time_length>
# Example: python 7_clustering.py SYK D:\Infrasound\db\th+fft D:\Infrasound\db\band_power 3600
```

**Outputs**:
- `{PLACE}_band_power.npy`: Band power matrix
- `{PLACE}_bp_labeled.csv`: Band power with cluster labels
- `{PLACE}(freq_band)_cluster_plot.png`: Cluster scatter plot
- `{PLACE}(freq_band)_band_power.png`: Time series with clusters

#### [7b_clustering.py](7b_clustering.py)
DBSCAN clustering for density-based pattern detection:

**Advantages over K-means**:
- No need to specify number of clusters
- Can identify noise points (label = -1)
- Better for non-spherical cluster shapes
- Automatically detects outliers

**Parameters**:
- `eps=0.5`: Maximum distance between samples
- `min_samples=10`: Minimum samples for core point
- `metric='euclidean'`: Distance metric

**Use Cases**:
- Detecting transient volcanic events
- Identifying anomalous activity periods
- Separating background noise from signals

## Signal Processing Methodology

### Frequency Band Selection

The analysis focuses on specific frequency bands relevant to volcanic monitoring:

| Band (Hz) | Phenomena | Possible Significance |
|-----------|-----------|--------------|
| 0.001-0.01 | Long-period oscillations | Magma movement, chamber resonance |
| 0.01-0.1 | Very long-period signals | Deep volcanic processes |
| 0.1-1 | Microbaroms | Ocean wave interactions, meteorological |
| 1-10 | Volcanic tremor | Ongoing volcanic activity, explosions |
| 6-8 | Target band | Specific volcanic signature |
| 9-10 | High-frequency tremor | Surface activity, degassing |
| 10-20 | High-frequency events | Near-surface processes |

### Feature Extraction

**Band Power Calculation**:
```python
# Power Spectral Density calculation
PSD = np.abs(FFT)**2 / N

# Band Power summation over frequency range
Band_Power = np.sum(PSD[(freq >= f_low) & (freq <= f_high)])
```

**Normalization**:
- Z-score normalization applied to remove station-specific biases
- Handles missing data using NaN masking
- Mean imputation for temporal gaps

**Peak Detection**:
- `order=3` for peak detection (local maximum)
- `order=15` for valley detection (local minimum)
- Helps identify transient events and activity cycles

## Clustering Strategy

### K-Means Clustering (7_clustering.py)

**Objectives**:
- Partition continuous infrasound data into distinct states
- Identify background, normal, and elevated activity periods
- Track temporal evolution of volcanic behavior

**Process**:
1. Extract band power features for all sensors
2. Normalize across time and stations
3. Apply K-means with k=3 to k=10 clusters
4. Validate using silhouette score and domain knowledge
5. Visualize cluster temporal distribution

### DBSCAN Clustering (7b_clustering.py)

**Objectives**:
- Detect outlier events (potential eruptions, explosions)
- Identify dense regions of similar activity
- Avoid assuming spherical cluster shapes

**Advantages**:
- Robust to noise and outliers
- No predefined cluster count
- Captures irregular activity patterns

## Dependencies

```python
numpy              # Numerical computing
scipy              # Signal processing (FFT, filtering, peak detection)
matplotlib         # Visualization
scikit-learn       # Machine learning (KMeans, DBSCAN, preprocessing)
```

## File Structure

```
.
├── 6_band_power.py          # Basic band power extraction
├── 6b_band_power.py         # Flexible band power analysis
├── 6c_band_power.py         # Multi-station band power with visualization
├── 6c_psd.py                # Welch PSD analysis
├── 6d_.py                   # Alternative band power implementation
├── 7_clustering.py          # K-means temporal clustering
├── 7b_clustering.py         # DBSCAN clustering
├── 7_selectplot.py          # Cluster visualization and selection
├── 8_split.py               # Data organization by clusters
├── common.py                # Utility functions
└── 報告/
    └── 大屯火山群高密度次聲波資料分析_林奕丞_期末海報.pdf  # Project poster
```

## Key Findings (Summary from Poster)

Based on the analysis of the Tatun Volcano Group infrasound data:

1. **Temporal Patterns**: Distinct activity patterns identified through clustering analysis
2. **Frequency Characteristics**: Multiple frequency bands showing different behaviors
3. **Spatial Distribution**: Variation in signals across different sensor locations
4. **Anomaly Detection**: DBSCAN effectively identifies transient events

## Workflow Example

Complete analysis pipeline:

```bash
# Step 1: Calculate band power for all sensors (SYK station, 1-hour windows)
python 6c_band_power.py SYK D:\Infrasound\db\th+fft D:\Infrasound\db\band_power 3600

# Step 2: Apply K-means clustering (3 clusters for activity states)
python 7_clustering.py SYK D:\Infrasound\db\th+fft D:\Infrasound\db\band_power 3600

# Step 3: Alternative DBSCAN clustering for anomaly detection
python 7b_clustering.py SYK D:\Infrasound\db\th+fft D:\Infrasound\db\band_power 3600

# Step 4: Organize results by clusters
python 8_split.py SYK D:\Infrasound\db\th+fft D:\Infrasound\db\band_power 3600

# Step 5: Generate selected visualizations
python 7_selectplot.py SYK D:\Infrasound\db\th+fft D:\Infrasound\db\band_power 3600
```

## Configuration Parameters

### Time Windows
- `time_length=3600`: 1-hour analysis windows
- `time_length=1800`: 30-minute windows
- `time_length=900`: 15-minute windows

### Frequency Bands
Configurable in each script:
```python
band = [[0.1, 1], [1, 10], [10, 20]]  # Standard bands
band = [[6, 8], [9, 10]]              # Target volcanic signatures
```

### Clustering Parameters
- **K-means**: `num_cluster=3` to `num_cluster=10`
- **DBSCAN**: `eps=0.5`, `min_samples=10`

## Output Files

### Band Power Analysis
- `{PLACE}_band_power.npy`: NumPy array of band power values
- `{PLACE}_band_power.csv`: CSV export for external analysis
- `{PLACE}_peaks_save.csv`: Detected peaks and valleys
- Visualization PNG files

### Clustering Results
- `{PLACE}_bp_labeled.csv`: Band power with cluster labels
- `{PLACE}_cluster_plot.png`: Cluster distribution scatter plot
- `{PLACE}({freq_band})_band_power.png`: Time series with cluster coloring

## Future Work

- Multi-station correlation analysis
- Machine learning classification of event types
- Real-time monitoring implementation
- Integration with seismic data
- Automated anomaly detection system

## Publications

Lin, Y.-C. (2nd-author), "Classification of Infrasonic Signals of Tatun Volcano Group with Unsupervised Machine Learning," *16th World Congress on Computational Mechanics and 4th Pan American Congress on Computational Mechanics (WCCM-PANACM 2024)*, Vancouver, Canada, July 21-26, 2024, Paper No. W242098.

## References

For detailed methodology and results, please refer to:
- Project Poster (Chinese): [大屯火山群高密度次聲波資料分析_林奕丞_期末海報.pdf](報告/大屯火山群高密度次聲波資料分析_林奕丞_期末海報.pdf)

## Author

Lin Yi-Cheng (林奕丞), [National Center for Research on Earthquake Engineering (NCREE) AI Research Center](https://www.aiengineer.tw/o-home.html)

## License

Note: Raw infrasound data is proprietary and not included in this repository due to data sharing restrictions.

**Acknowledgments**: This analysis was conducted as part of volcanic monitoring research for the Tatun Volcano Group in northern Taiwan.
