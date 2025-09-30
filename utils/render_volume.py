# render_volume_plotly.py

def render_volume(volume_data):
    """
    Render a 3D volume using Plotly's volume rendering.
    
    Parameters:
        volume_data (numpy.ndarray): A 3D numpy array with shape 
                                     (height, width, n_depths)
    """
    import numpy as np
    import plotly.graph_objects as go
    import plotly.colors as pc

    def nonlinear_colorscale(base_colormap, gamma=1.0):
        """
        Create a custom colorscale by applying a nonlinear mapping to the 
        normalized values of a built-in colormap.
        
        Parameters:
          base_colormap: list of color strings (e.g., from pc.sequential.YlGnBu)
          gamma: float
                 - gamma = 1.0 gives a linear mapping.
                 - gamma < 1.0 gives a mapping that is slower at low values then faster.
                 - gamma > 1.0 gives a mapping that is faster at low values then slower.
        
        Returns:
          custom_colorscale: list of [normalized_value, color] pairs.
        """
        n_colors = len(base_colormap)
        custom_colorscale = []
        for i, color in enumerate(base_colormap):
            # Original normalized value (linear spacing)
            v_linear = i / (n_colors - 1)
            # Apply the nonlinear mapping: f(v) = v^(1/gamma)
            v_nonlinear = v_linear ** (1.0 / gamma)
            custom_colorscale.append([v_nonlinear, color])
        return custom_colorscale

    # Determine the dimensions from the volume data.
    # Note: volume_data.shape returns (height, width, n_depths)
    height, width, n_depths = volume_data.shape

    # Create coordinate arrays for each axis.
    # We use meshgrid to create X, Y, Z arrays.
    # Note: we use indexing='ij' so that the first dimension corresponds to x.
    # Since our volume_data is (height, width, n_depths) and typically image rows
    # correspond to y and columns correspond to x, we swap the first two axes.
    X, Y, Z = np.meshgrid(np.arange(width), np.arange(height), np.arange(n_depths), indexing='xy')
    
    # Flatten the coordinate arrays and the volume data.
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    values = volume_data.flatten()
    opacityscale = [
        [0.0, 0.0],   # When the normalized data value is 0, opacity is 0.2
        [0.01, 0.5],   
        [1.0, 0.8]    
    ]

    base_colors = pc.sequential.YlOrRd
    # base_colors = pc.sequential.YlGnBu 
    gamma = 0.75  # Change this parameter: try 1 for linear, <1 for slow-then-fast, >1 for fast-then-slow
    custom_colorscale = nonlinear_colorscale(base_colors, gamma)

    # Create a Plotly volume trace.
    fig = go.Figure(data=go.Volume(
        x=x_flat,
        y=y_flat,
        z=z_flat,
        value=values,
        isomin=float(np.min(volume_data)),
        isomax=float(np.max(volume_data)),
        opacityscale=opacityscale,  # Map data values to opacity
        opacity=0.1,              # Overall opacity factor (adjust as needed)
        surface_count=40,         # Number of isosurfaces
        colorscale=custom_colorscale          # Use colormap
    ))
    
    # Update the layout with axis titles.
    fig.update_layout(
        title="3D Volume Rendering using Plotly",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Depth"
        )
    )
    
    # Display the interactive plot.
    fig.show()
