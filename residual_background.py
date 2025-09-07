'''
Núcleo 1 - Residual background phase variance: Métricas convencionales. Se mide phase flatness in
background (object-free) regions of the reconstructed phase image. En estas, hay que hacer un
segmentador, para saber cuáles son las zonas de fondo en el objeto.
Métricas en este grupo:
- Standard deviation (STD) or Mean Absolute Deviation (MAD) (o las dos).
- RMS.
- Peak-to-Valley (P–V) Value in Background.
- Background phase tilt/Curvature residuals (esta involucra medir los coeficientes de legenre/zernike para el mapa de fase resultante, y con ellos, saber cuáles son los residuales)
- Full Width at Half Maximum (FWHM) of the phase histogram (esta es muy buena cuando hay bastante fondo, yo diría, la mejor. Me cuentas si no sabes cómo medirla.)
- Spatial frequency content of background (la misma que implementaste para el paper del VortexLegendre, pero para el fondo solamente).
- Entropy of background phase map (creo que ya la tienes, pero para el bakground).
'''

# Libraries
import numpy as np



def spatial_frequency(self, img):
    gray = np.array(img.convert("L")).astype(np.float64)
    M, N = gray.shape
    RF = np.sqrt(np.sum((gray[:, 1:] - gray[:, :-1]) ** 2) / (M * N))
    CF = np.sqrt(np.sum((gray[1:, :] - gray[:-1, :]) ** 2) / (M * N))
    SF = np.sqrt(RF ** 2 + CF ** 2)

    return SF


def entropy_background(self, img):
    gray_img = img.convert("L")
    hist = gray_img.histogram()
    hist = np.array(hist) / np.sum(hist)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return entropy


