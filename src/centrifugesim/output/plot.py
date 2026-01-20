import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def plot_field(data, geom, title, cbar_label, 
               log_scale=False, divergent=False, vmin=None, vmax=None, filename=None):
    """
    Plots a 2D field with automatic masking and formatting.
    
    Parameters:
    - divergent (bool): If True, uses a Red-White-Blue colormap and centers 
      the color scale at 0 by making vmin/vmax symmetric.
    """
    # 1. Create copy and apply mask
    data_plot = np.copy(data)
    # Handle the case where geom.mask might be boolean or binary
    data_plot[geom.mask == 0] = np.nan
    
    # 2. Setup Figure
    plt.figure(figsize=(11, 5))
    
    # 3. Configure Normalization, Limits, and Colormap
    norm = None
    plot_kwargs = {} 
    
    # Default colormap is plasma, but switch to seismic (Red-White-Blue) if divergent
    cmap = 'seismic' if divergent else 'plasma'
    
    if log_scale:
        # Log scale logic (Divergent logic rarely applies to Log scale, 
        # but we keep user vmin/vmax if provided)
        if vmin is None: vmin = np.nanmin(data_plot)
        if vmax is None: vmax = np.nanmax(data_plot)
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        # Linear scale logic
        if divergent:
            # Calculate the absolute maximum to ensure symmetry around 0
            abs_max = np.nanmax(np.abs(data_plot))
            
            # Only override vmin/vmax if the user didn't provide specific ones
            if vmin is None: vmin = -abs_max
            if vmax is None: vmax = abs_max
            
        # Pass limits to pcolormesh kwargs
        if vmin is not None: plot_kwargs['vmin'] = vmin
        if vmax is not None: plot_kwargs['vmax'] = vmax
        
    # 4. Plot
    # Pass the selected 'cmap' here
    pcm = plt.pcolormesh(geom.Z, geom.R, data_plot, cmap=cmap, shading='auto',
                         norm=norm, **plot_kwargs)
    
    # 5. Decoration
    cbar = plt.colorbar(pcm, label=cbar_label)
    plt.xlabel('z (m)')
    plt.ylabel('r (m)')
    plt.title(title)
    plt.tight_layout()
    
    # 6. Save or Show
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved: {filename}")
    
    plt.show()
    plt.close()