import os
import numpy as np
from dolfinx import io, fem
from mpi4py import MPI
import ufl

OUTPUT_DIR="outputs"

def _ensure_dir(comm):
    """Safely create the output directory in a parallel MPI environment."""
    if comm.rank == 0:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Block all processes until the master process has finished creating the folder
    comm.barrier()

def save_solution_to_xdmf(uh, domain, filename_base="solution"):
    """Save the complex solution as real, imaginary, AND absolute amplitude fields."""

    _ensure_dir(domain.comm)
    
    uh_real = fem.Function(uh.function_space)
    uh_real.name = "Real_Part"
    
    uh_imag = fem.Function(uh.function_space)
    uh_imag.name = "Imaginary_Part"
    
    uh_amp = fem.Function(uh.function_space)
    uh_amp.name = "Absolute_Amplitude"

    # 1. Real and Imaginary are LINEAR operations. We can safely extract the raw arrays.
    uh_real.x.array[:] = np.real(uh.x.array)
    uh_imag.x.array[:] = np.imag(uh.x.array)
    
    # 2. Amplitude is NON-LINEAR. We must mathematically project it across the polynomials.
    # ufl.inner(uh, uh) automatically computes u * conj(u) = |u|^2
    amp_expr = fem.Expression(ufl.sqrt(ufl.inner(uh, uh)), uh.function_space.element.interpolation_points)
    uh_amp.interpolate(amp_expr)

    # Prepend the output directory to the filenames
    path_real = os.path.join(OUTPUT_DIR, f"{filename_base}_real.bp")
    path_imag = os.path.join(OUTPUT_DIR, f"{filename_base}_imag.bp")
    path_amp  = os.path.join(OUTPUT_DIR, f"{filename_base}_amp.bp")

    try:
        # Save all three natively in the DG space using VTX
        with io.VTXWriter(domain.comm, path_real, [uh_real], engine="BP4") as vtx:
            vtx.write(0.0)
        with io.VTXWriter(domain.comm, path_imag, [uh_imag], engine="BP4") as vtx:
            vtx.write(0.0)
        with io.VTXWriter(domain.comm, path_amp, [uh_amp], engine="BP4") as vtx:
            vtx.write(0.0)
            
        if domain.comm.rank == 0:
            print(f"Saved Real, Imaginary, and Amplitude fields to {OUTPUT_DIR}/")
    except Exception as e:
        print(f"VTXWriter failed: {e}")

def save_error_to_xdmf(uh, u_exact, domain, filename_base="error"):
    """Save absolute error |uh - u_exact|."""

    _ensure_dir(domain.comm)

    error = fem.Function(uh.function_space)
    error.name = "Absolute_Error"

    # Error magnitude is also NON-LINEAR. Calculate using UFL.
    err_diff = uh - u_exact
    err_expr = fem.Expression(ufl.sqrt(ufl.inner(err_diff, err_diff)), uh.function_space.element.interpolation_points)
    error.interpolate(err_expr)

    path_err = os.path.join(OUTPUT_DIR, f"{filename_base}.bp")

    try:
        with io.VTXWriter(domain.comm, path_err, [error], engine="BP4") as vtx:
            vtx.write(0.0)
        if domain.comm.rank == 0:
            print(f"Saved error field to {path_err}")
    except Exception as e:
        print(f"VTXWriter failed: {e}")

def export_animation(uh, domain, k, filename="wave_animation.bp"):
    """
    Exports a time-domain animation natively in the DG space.
    Because it remains in DG, ParaView will render the actual discontinuities.
    """
    _ensure_dir(domain.comm)
    path_anim = os.path.join(OUTPUT_DIR, filename)

    if domain.comm.rank == 0:
        print(f"Generating Animation Frames in {path_anim}...")

    # Create a container to hold each frame of the video
    uh_animated = fem.Function(uh.function_space)
    uh_animated.name = "Acoustic_Wave"

    # Time-harmonic animation parameters
    nFrames = 60
    omega = k * 1.0  # Wavenumber * Wave Speed
    dt = (2 * np.pi / omega) / (nFrames / 2.0)

    try:
        # Open the VTX writer and inject frames one by one
        with io.VTXWriter(domain.comm, path_anim, [uh_animated], engine="BP4") as vtx:
            for i in range(nFrames):
                t = i * dt
                
                # Apply the phase shift: Real( u * e^{-i * omega * t} )
                # Note: Phase shift is a linear operation, so doing it directly on the array is 100% correct!
                current_wave = np.real(uh.x.array * np.exp(-1j * omega * t))
                
                # Assign to the complex array (imaginary parts become 0)
                uh_animated.x.array[:] = current_wave.astype(uh.x.array.dtype)
                vtx.write(t)
                
        if domain.comm.rank == 0:
            print("Animation export complete!")
    except Exception as e:
        print(f"Animation export failed: {e}")