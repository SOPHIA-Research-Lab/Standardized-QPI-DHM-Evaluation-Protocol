import wx

def create(parent_frame, notebook):
    analysis_menu = wx.Menu()
    module1_menu = wx.Menu()
    module2_menu = wx.Menu()
    module3_menu = wx.Menu()
    module4_menu = wx.Menu()

    parent_frame.computational_complexity_id = wx.NewIdRef()
    module4_menu.Append(parent_frame.computational_complexity_id, "Launch Complexity Analyzer")

    # ==========================
    # STD
    # ==========================
    std_submenu = wx.Menu()
    parent_frame.std_background_unwrapped_id = wx.NewIdRef()
    parent_frame.std_background_id = wx.NewIdRef()
    parent_frame.std_background_zones_id = wx.NewIdRef()

    std_submenu.Append(parent_frame.std_background_unwrapped_id, "STD - Unwrapped Background")
    std_submenu.Append(parent_frame.std_background_id, "STD - Background")
    std_submenu.Append(parent_frame.std_background_zones_id, "STD - Background Zones")

    module1_menu.AppendSubMenu(std_submenu, "Standard Deviation (STD)")

    # ==========================
    # MAD
    # ==========================
    mad_submenu = wx.Menu()
    parent_frame.mad_unwrapped_id = wx.NewIdRef()
    parent_frame.mad_background_id = wx.NewIdRef()
    parent_frame.mad_background_zones_id = wx.NewIdRef()

    mad_submenu.Append(parent_frame.mad_unwrapped_id, "MAD - Unwrapped Background")
    mad_submenu.Append(parent_frame.mad_background_id, "MAD - Background")
    mad_submenu.Append(parent_frame.mad_background_zones_id, "MAD - Background Zones")

    module1_menu.AppendSubMenu(mad_submenu, "Mean Absolute Deviation (MAD)")

    # ==========================
    # RMS
    # ==========================
    rms_submenu = wx.Menu()
    parent_frame.rms_unwrapped_id = wx.NewIdRef()
    parent_frame.rms_background_id = wx.NewIdRef()
    parent_frame.rms_background_zones_id = wx.NewIdRef()

    rms_submenu.Append(parent_frame.rms_unwrapped_id, "RMS - Unwrapped Background")
    rms_submenu.Append(parent_frame.rms_background_id, "RMS - Background")
    rms_submenu.Append(parent_frame.rms_background_zones_id, "RMS - Background Zones")

    module1_menu.AppendSubMenu(rms_submenu, "Root Mean Square (RMS)")

    # ==========================
    # PV
    # ==========================
    pv_submenu = wx.Menu()
    parent_frame.pv_unwrapped_id = wx.NewIdRef()
    parent_frame.pv_background_id = wx.NewIdRef()
    parent_frame.pv_background_zones_id = wx.NewIdRef()

    pv_submenu.Append(parent_frame.pv_unwrapped_id, "PV - Unwrapped Background")
    pv_submenu.Append(parent_frame.pv_background_id, "PV - Background")
    pv_submenu.Append(parent_frame.pv_background_zones_id, "PV - Background Zones")

    module1_menu.AppendSubMenu(pv_submenu, "Peak-to-Valley (PV)")

    # ==========================
    # FWHM
    # ==========================
    fwhm_submenu = wx.Menu()
    parent_frame.fwhm_unwrapped_id = wx.NewIdRef()
    parent_frame.fwhm_background_id = wx.NewIdRef()
    parent_frame.fwhm_background_zones_id = wx.NewIdRef()

    fwhm_submenu.Append(parent_frame.fwhm_unwrapped_id, "FWHM - Unwrapped Background")
    fwhm_submenu.Append(parent_frame.fwhm_background_id, "FWHM - Background")
    fwhm_submenu.Append(parent_frame.fwhm_background_zones_id, "FWHM - Background Zones")

    module1_menu.AppendSubMenu(fwhm_submenu, "Full Width at Half Maximum (FWHM)")

    # ==========================
    # Entropy
    # ==========================
    entropy_submenu = wx.Menu()
    parent_frame.entropy_unwrapped_id = wx.NewIdRef()
    parent_frame.entropy_background_id = wx.NewIdRef()
    parent_frame.entropy_background_zones_id = wx.NewIdRef()

    entropy_submenu.Append(parent_frame.entropy_unwrapped_id, "Entropy - Unwrapped Background")
    entropy_submenu.Append(parent_frame.entropy_background_id, "Entropy - Background")
    entropy_submenu.Append(parent_frame.entropy_background_zones_id, "Entropy - Background Zones")

    module1_menu.AppendSubMenu(entropy_submenu, "Entropy")

    # ==========================
    # Legendre Coefficients
    # ==========================
    legendre_submenu = wx.Menu()
    parent_frame.legendre_unwrapped_id = wx.NewIdRef()
    parent_frame.legendre_background_id = wx.NewIdRef()
    parent_frame.legendre_background_zones_id = wx.NewIdRef()

    legendre_submenu.Append(parent_frame.legendre_unwrapped_id, "Legendre - Unwrapped Background")
    legendre_submenu.Append(parent_frame.legendre_background_id, "Legendre - Background")
    legendre_submenu.Append(parent_frame.legendre_background_zones_id, "Legendre - Background Zones")

    module1_menu.AppendSubMenu(legendre_submenu, "Legendre Coefficients")


    # ==========================
    # General option: All metrics Mod1
    # ==========================
    parent_frame.all_id = wx.NewIdRef()
    module1_menu.AppendSeparator()
    module1_menu.Append(parent_frame.all_id, "Calculate All Metrics")

    # ==========================
    # Global Phase Distortion Metrics (Module 2)
    # ==========================
    
    # Maximum-Minus-Minimum 
    mmm_submenu = wx.Menu()
    parent_frame.mmm_global_id = wx.NewIdRef()
    parent_frame.mmm_global_unwrapped_id = wx.NewIdRef()
    mmm_submenu.Append(parent_frame.mmm_global_id, "Maximum-Minus-Minimum")
    mmm_submenu.Append(parent_frame.mmm_global_unwrapped_id, "Maximum-Minus-Minimum - Unwrapped")
    module2_menu.AppendSubMenu(mmm_submenu, "Maximum-Minus-Minimum")


    # Global Phase Gradient 
    gradient_submenu = wx.Menu()
    parent_frame.gradient_global_id = wx.NewIdRef()
    parent_frame.gradient_global_unwrapped_id = wx.NewIdRef()
    gradient_submenu.Append(parent_frame.gradient_global_id, "Phase Gradient")
    gradient_submenu.Append(parent_frame.gradient_global_unwrapped_id, "Phase Gradient - Unwrapped")
    module2_menu.AppendSubMenu(gradient_submenu, "Global Phase Gradient")

    # TSM submenu
    tsm_submenu = wx.Menu()
    parent_frame.tsm_global_id = wx.NewIdRef()
    parent_frame.tsm_global_unwrapped_id = wx.NewIdRef()
    tsm_submenu.Append(parent_frame.tsm_global_id, "TSM")
    tsm_submenu.Append(parent_frame.tsm_global_unwrapped_id, "TSM - Unwrapped")
    module2_menu.AppendSubMenu(tsm_submenu, "TSM")

    # Phase Curvature Coefficients submenu
    curvature_submenu = wx.Menu()
    parent_frame.curvature_global_id = wx.NewIdRef()
    parent_frame.curvature_global_unwrapped_id = wx.NewIdRef()
    curvature_submenu.Append(parent_frame.curvature_global_id, "Phase Curvature")
    curvature_submenu.Append(parent_frame.curvature_global_unwrapped_id, "Phase Curvature - Unwrapped")
    module2_menu.AppendSubMenu(curvature_submenu, "Phase Curvature")

    # Laplacian Energy submenu
    laplacian_submenu = wx.Menu()
    parent_frame.laplacian_global_id = wx.NewIdRef()
    parent_frame.laplacian_global_unwrapped_id = wx.NewIdRef()
    laplacian_submenu.Append(parent_frame.laplacian_global_id, "Laplacian Energy")
    laplacian_submenu.Append(parent_frame.laplacian_global_unwrapped_id, "Laplacian Energy - Unwrapped")
    module2_menu.AppendSubMenu(laplacian_submenu, "Laplacian Energy")

    # Spatial Frequency submenu
    spatial_freq_submenu = wx.Menu()
    parent_frame.spatial_freq_global_id = wx.NewIdRef()
    parent_frame.spatial_freq_global_unwrapped_id = wx.NewIdRef()
    spatial_freq_submenu.Append(parent_frame.spatial_freq_global_id, "Spatial Frequency")
    spatial_freq_submenu.Append(parent_frame.spatial_freq_global_unwrapped_id, "Spatial Frequency - Unwrapped")
    module2_menu.AppendSubMenu(spatial_freq_submenu, "Spatial Frequency")

    # Global Entropy submenu
    entropy_submenu = wx.Menu()
    parent_frame.global_entropy_global_id = wx.NewIdRef()
    parent_frame.global_entropy_global_unwrapped_id = wx.NewIdRef()
    entropy_submenu.Append(parent_frame.global_entropy_global_id, "Entropy")
    entropy_submenu.Append(parent_frame.global_entropy_global_unwrapped_id, "Entropy - Unwrapped")
    module2_menu.AppendSubMenu(entropy_submenu, "Global Entropy")

    # Sharpness/Contrast submenu
    sharpness_submenu = wx.Menu()
    parent_frame.sharpness_global_id = wx.NewIdRef()
    parent_frame.sharpness_global_unwrapped_id = wx.NewIdRef()
    sharpness_submenu.Append(parent_frame.sharpness_global_id, "Sharpness/Contrast")
    sharpness_submenu.Append(parent_frame.sharpness_global_unwrapped_id, "Sharpness/Contrast - Unwrapped")
    module2_menu.AppendSubMenu(sharpness_submenu, "Sharpness/Contrast")

    # General option: All metrics Mod2
    parent_frame.allM2_id = wx.NewIdRef()
    module2_menu.AppendSeparator()
    module2_menu.Append(parent_frame.allM2_id, "Calculate All Metrics")

    # ----------------------------------------------------------
    # Add modules to main menu
    # ----------------------------------------------------------
    analysis_menu.AppendSubMenu(module1_menu, "Residual Background Phase Variance")
    analysis_menu.AppendSubMenu(module2_menu, "Global Phase Distortion Metrics")
    analysis_menu.AppendSubMenu(module3_menu, "Ground-Truth Comparisons")
    analysis_menu.AppendSubMenu(module4_menu, "Computational Complexity")

    # Save references (useful for enable/disable)
    parent_frame.module1_menu = module1_menu
    parent_frame.module2_menu = module2_menu
    parent_frame.std_submenu = std_submenu
    parent_frame.mad_submenu = mad_submenu
    parent_frame.rms_submenu = rms_submenu
    parent_frame.pv_submenu = pv_submenu
    parent_frame.fwhm_submenu = fwhm_submenu
    parent_frame.entropy_submenu = entropy_submenu
    parent_frame.legendre_submenu = legendre_submenu
    


    # ----------------------------------------------------------
    # Bind events - Module 1 (Residual Background Phase Variance)
    # ----------------------------------------------------------
    # STD
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_std_background_unwrapped, id=parent_frame.std_background_unwrapped_id)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_std_background, id=parent_frame.std_background_id)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_std_background_zones, id=parent_frame.std_background_zones_id)

    # MAD
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_mad_background_unwrapped, id=parent_frame.mad_unwrapped_id)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_mad_background, id=parent_frame.mad_background_id)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_mad_background_zones, id=parent_frame.mad_background_zones_id)

    # RMS
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_rms_background_unwrapped, id=parent_frame.rms_unwrapped_id)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_rms_background, id=parent_frame.rms_background_id)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_rms_background_zones, id=parent_frame.rms_background_zones_id)

    # PV
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_pv_background_unwrapped, id=parent_frame.pv_unwrapped_id)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_pv_background, id=parent_frame.pv_background_id)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_pv_background_zones, id=parent_frame.pv_background_zones_id)

    # FWHM
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_fwhm_background_unwrapped, id=parent_frame.fwhm_unwrapped_id)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_fwhm_background, id=parent_frame.fwhm_background_id)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_fwhm_background_zones, id=parent_frame.fwhm_background_zones_id)

    # Entropy
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_entropy_background_unwrapped, id=parent_frame.entropy_unwrapped_id)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_entropy_background, id=parent_frame.entropy_background_id)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_entropy_background_zones, id=parent_frame.entropy_background_zones_id)
    
    # Legendre Coefficients
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_legendre_background_unwrapped, id=parent_frame.legendre_unwrapped_id)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_legendre_background, id=parent_frame.legendre_background_id)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_legendre_background_zones, id=parent_frame.legendre_background_zones_id)

    # All metrics
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_all_metrics, id=parent_frame.all_id)

    # ----------------------------------------------------------
    # Bind events - Module 2 (Global Phase Distortion Metrics)
    # ----------------------------------------------------------
    # Maximum-Minus-Minimum
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_mmm_global, id=parent_frame.mmm_global_id)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_mmm_global_unwrapped, id=parent_frame.mmm_global_unwrapped_id)
   

    # Global Phase Gradient
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_gradient_global, id=parent_frame.gradient_global_id)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_gradient_global_unwrapped, id=parent_frame.gradient_global_unwrapped_id)
    
    # TSM
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_tsm_global, id=parent_frame.tsm_global_id)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_tsm_global_unwrapped, id=parent_frame.tsm_global_unwrapped_id)

    # Phase Curvature Coefficients
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_curvature_global, id=parent_frame.curvature_global_id)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_curvature_global_unwrapped, id=parent_frame.curvature_global_unwrapped_id)

    # Laplacian Energy
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_laplacian_global, id=parent_frame.laplacian_global_id)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_laplacian_global_unwrapped, id=parent_frame.laplacian_global_unwrapped_id)

    # Spatial Frequency
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_spatial_freq_global, id=parent_frame.spatial_freq_global_id)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_spatial_freq_global_unwrapped, id=parent_frame.spatial_freq_global_unwrapped_id)


    # Global Entropy
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_global_entropy_global, id=parent_frame.global_entropy_global_id)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_global_entropy_global_unwrapped, id=parent_frame.global_entropy_global_unwrapped_id)

     # Sharpness/Contrast
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_sharpness_global, id=parent_frame.sharpness_global_id)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_sharpness_global_unwrapped, id=parent_frame.sharpness_global_unwrapped_id)

    # Legendre Coefficients
    # parent_frame.Bind(wx.EVT_MENU, parent_frame.on_legendre_global, id=parent_frame.legendre_global_id)
    # parent_frame.Bind(wx.EVT_MENU, parent_frame.on_legendre_global_unwrapped, id=parent_frame.legendre_global_unwrapped_id)

    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_computational_complexity, id=parent_frame.computational_complexity_id)

    # All metrics
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_all_global_metrics_M2, id=parent_frame.allM2_id)
    # ==========================
    # Ground-Truth Comparisons (Module 3)
    # ==========================

    # SSIM submenu
    ssim_submenu = wx.Menu()
    parent_frame.ssim_comparison_id = wx.NewIdRef()
    parent_frame.ssim_comparison_unwrapped_id = wx.NewIdRef()
    ssim_submenu.Append(parent_frame.ssim_comparison_id, "Structural Similarity (SSIM)")
    ssim_submenu.Append(parent_frame.ssim_comparison_unwrapped_id, "Structural Similarity (SSIM) Unwrapped")
    module3_menu.AppendSubMenu(ssim_submenu, "SSIM (Structural Similarity)")

    # MSE submenu
    mse_submenu = wx.Menu()
    parent_frame.mse_comparison_id = wx.NewIdRef()
    parent_frame.mse_comparison_unwrapped_id = wx.NewIdRef()
    mse_submenu.Append(parent_frame.mse_comparison_id, "Mean Squared Error (MSE)")
    mse_submenu.Append(parent_frame.mse_comparison_unwrapped_id, "Mean Squared Error (MSE) Unwrapped")
    module3_menu.AppendSubMenu(mse_submenu, "MSE (Mean Squared Error)")

    # PSNR submenu
    psnr_submenu = wx.Menu()
    parent_frame.psnr_comparison_id = wx.NewIdRef()
    parent_frame.psnr_comparison_unwrapped_id = wx.NewIdRef()
    psnr_submenu.Append(parent_frame.psnr_comparison_id, "Peak Signal-to-Noise Ratio (PSNR)")
    psnr_submenu.Append(parent_frame.psnr_comparison_unwrapped_id, "Peak Signal-to-Noise Ratio (PSNR) Unwrapped")
    module3_menu.AppendSubMenu(psnr_submenu, "PSNR")

    # Opción para cargar Ground-Truth
    module3_menu.AppendSeparator()
    parent_frame.load_ground_truth_id = wx.NewIdRef()
    module3_menu.Append(parent_frame.load_ground_truth_id, "Load Ground-Truth Image...")

    # ----------------------------------------------------------
    # Bind events - Module 3 (Ground-Truth Comparisons)
    # ----------------------------------------------------------
    # SSIM
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_ssim_comparison, id=parent_frame.ssim_comparison_id)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_ssim_comparison_unwrapped, id=parent_frame.ssim_comparison_unwrapped_id)

    # MSE
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_mse_comparison, id=parent_frame.mse_comparison_id)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_mse_comparison_unwrapped, id=parent_frame.mse_comparison_unwrapped_id)

    # PSNR
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_psnr_comparison, id=parent_frame.psnr_comparison_id)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_psnr_comparison_unwrapped, id=parent_frame.psnr_comparison_unwrapped_id)

    # Load Ground-Truth
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_load_ground_truth, id=parent_frame.load_ground_truth_id)


    return analysis_menu