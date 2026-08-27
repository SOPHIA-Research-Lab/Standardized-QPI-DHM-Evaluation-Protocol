
open("C:\Users\gaezn\Documents\Repositorios\Github\Standardized-QPI-DHM-Evaluation-Protocol\My_app\Samples Hologram\Telecentric\T_Probiotics_20x_632_3.75.bmp");
run("Gaussian Blur...", "sigma=5");
saveAs("Tiff", "C:\Users\gaezn\Documents\Repositorios\Github\Standardized-QPI-DHM-Evaluation-Protocol\My_app\Samples Hologram\resultado_filtrado.tif");
close();
print("Proceso completado");
