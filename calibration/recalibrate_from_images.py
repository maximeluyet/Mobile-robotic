import numpy as np
import cv2
import os
import glob
from datetime import datetime

# Configuration
INPUT_DIRECTORY = 'calibration_output'  # Dossier contenant les images déjà prises
OUTPUT_DIRECTORY = 'calibration_output'  # Dossier pour les fichiers de sortie
CHESSBOARD_SIZE = (9, 6)  # Nombre de coins intérieurs du damier (colonnes, lignes)
SQUARE_SIZE = 2.5  # Taille d'un carré en cm (pour mesures réelles)

# Liste de tailles alternatives à essayer si la détection échoue
ALTERNATIVE_SIZES = [
    (9, 6), (6, 9), (8, 6), (6, 8), (7, 5), (5, 7),
    (10, 7), (7, 10), (8, 5), (5, 8), (7, 6), (6, 7)
]

def find_chessboard_in_image(gray_image, default_size=CHESSBOARD_SIZE):
    """
    Tente de détecter un damier dans une image en essayant plusieurs tailles.
    
    Args:
        gray_image: Image en niveaux de gris
        default_size: Taille par défaut à essayer en premier
    
    Returns:
        success: True si un damier a été détecté
        corners: Coordonnées des coins détectés
        detected_size: Taille du damier détecté
    """
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    
    # Essayer d'abord avec la taille par défaut
    ret, corners = cv2.findChessboardCorners(gray_image, default_size, flags)
    if ret:
        return True, corners, default_size
    
    # Essayer avec les tailles alternatives
    for size in ALTERNATIVE_SIZES:
        if size == default_size:
            continue
        ret, corners = cv2.findChessboardCorners(gray_image, size, flags)
        if ret:
            return True, corners, size
    
    return False, None, None

def recalibrate_from_existing_images():
    """
    Recalibre la caméra en utilisant les images déjà capturées dans le dossier.
    Met à jour les fichiers camera_matrix.txt et distortion_coefficients.txt.
    """
    print("\n" + "="*70)
    print("RECALIBRATION À PARTIR D'IMAGES EXISTANTES")
    print("="*70)
    
    # Vérifier que le dossier existe
    if not os.path.exists(INPUT_DIRECTORY):
        print(f"\n❌ Erreur: Le dossier '{INPUT_DIRECTORY}' n'existe pas")
        print("Assurez-vous d'avoir des images de calibration dans ce dossier.")
        return
    
    # Chercher toutes les images dans le dossier
    image_patterns = [
        os.path.join(INPUT_DIRECTORY, '*.jpg'),
        os.path.join(INPUT_DIRECTORY, '*.jpeg'),
        os.path.join(INPUT_DIRECTORY, '*.png')
    ]
    
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(pattern))
    
    if not image_files:
        print(f"\n❌ Erreur: Aucune image trouvée dans '{INPUT_DIRECTORY}'")
        print("Formats supportés: .jpg, .jpeg, .png")
        return
    
    print(f"\n✓ {len(image_files)} images trouvées dans '{INPUT_DIRECTORY}'")
    
    # Préparer les points 3D du damier
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE  # Mise à l'échelle avec la taille réelle
    
    # Tableaux pour stocker les points
    objpoints = []  # Points 3D dans l'espace réel
    imgpoints = []  # Points 2D dans le plan image
    
    img_shape = None
    detected_sizes = {}
    
    print("\nTraitement des images...")
    print("-" * 70)
    
    for idx, img_path in enumerate(image_files, 1):
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"⚠️  [{idx}/{len(image_files)}] {img_name}: Impossible de lire l'image")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if img_shape is None:
            img_shape = gray.shape[::-1]
        
        # Détecter le damier
        success, corners, detected_size = find_chessboard_in_image(gray, CHESSBOARD_SIZE)
        
        if success:
            # Suivre les tailles détectées
            if detected_size not in detected_sizes:
                detected_sizes[detected_size] = 0
            detected_sizes[detected_size] += 1
            
            # Préparer objp pour la taille détectée
            if detected_size != CHESSBOARD_SIZE:
                objp_custom = np.zeros((detected_size[0] * detected_size[1], 3), np.float32)
                objp_custom[:, :2] = np.mgrid[0:detected_size[0], 0:detected_size[1]].T.reshape(-1, 2)
                objp_custom = objp_custom * SQUARE_SIZE
                objpoints.append(objp_custom)
            else:
                objpoints.append(objp)
            
            # Raffiner la position des coins
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_refined)
            
            print(f"✓ [{idx}/{len(image_files)}] {img_name}: Damier {detected_size[0]}x{detected_size[1]} détecté")
        else:
            print(f"❌ [{idx}/{len(image_files)}] {img_name}: Damier non détecté")
    
    print("-" * 70)
    
    # Vérifier qu'on a assez d'images valides
    if len(objpoints) < 3:
        print(f"\n❌ Erreur: Pas assez d'images valides ({len(objpoints)} < 3)")
        print("La calibration nécessite au moins 3 images avec damier détecté")
        return
    
    print(f"\n✓ {len(objpoints)}/{len(image_files)} images valides pour la calibration")
    
    # Afficher les tailles de damier détectées
    if len(detected_sizes) > 1:
        print("\n⚠️  Attention: Plusieurs tailles de damier détectées:")
        for size, count in detected_sizes.items():
            print(f"   - {size[0]}x{size[1]}: {count} image(s)")
        print("   Cela peut affecter la précision de la calibration.")
    
    print("\n" + "="*70)
    print("CALCUL DE LA CALIBRATION")
    print("="*70)
    print("\nCalcul des paramètres intrinsèques de la caméra...")
    
    # Calibration de la caméra
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    
    # Créer le dossier de sortie si nécessaire
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    
    # Sauvegarder les résultats dans des fichiers texte
    camera_matrix_path = os.path.join(OUTPUT_DIRECTORY, 'camera_matrix.txt')
    distortion_path = os.path.join(OUTPUT_DIRECTORY, 'distortion_coefficients.txt')
    
    # Sauvegarder la matrice de caméra
    with open(camera_matrix_path, 'w') as f:
        f.write("# Camera Matrix (Intrinsic Parameters)\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Images used: {len(objpoints)}/{len(image_files)}\n")
        f.write(f"# RMS reprojection error: {ret:.6f} pixels\n")
        f.write("#\n")
        f.write("# Format: 3x3 matrix\n")
        f.write("# [[fx,  0, cx],\n")
        f.write("#  [ 0, fy, cy],\n")
        f.write("#  [ 0,  0,  1]]\n")
        f.write("#\n")
        for row in mtx:
            f.write(' '.join([f'{val:.6f}' for val in row]) + '\n')
    
    # Sauvegarder les coefficients de distorsion
    with open(distortion_path, 'w') as f:
        f.write("# Distortion Coefficients\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Images used: {len(objpoints)}/{len(image_files)}\n")
        f.write(f"# RMS reprojection error: {ret:.6f} pixels\n")
        f.write("#\n")
        f.write("# Format: [k1, k2, p1, p2, k3]\n")
        f.write("# k1, k2, k3: Radial distortion coefficients\n")
        f.write("# p1, p2: Tangential distortion coefficients\n")
        f.write("#\n")
        for val in dist[0]:
            f.write(f'{val:.6f}\n')
    
    # Afficher les résultats
    print("\n" + "="*70)
    print("RÉSULTATS DE LA CALIBRATION")
    print("="*70)
    print(f"\n✓ Erreur de reprojection RMS: {ret:.4f} pixels")
    print(f"✓ Images utilisées: {len(objpoints)}/{len(image_files)}")
    print(f"\n📋 Matrice de la caméra (paramètres intrinsèques):")
    print(f"   fx = {mtx[0, 0]:.2f} pixels (distance focale en x)")
    print(f"   fy = {mtx[1, 1]:.2f} pixels (distance focale en y)")
    print(f"   cx = {mtx[0, 2]:.2f} pixels (centre optique en x)")
    print(f"   cy = {mtx[1, 2]:.2f} pixels (centre optique en y)")
    print(f"\n📋 Coefficients de distorsion:")
    print(f"   k1 = {dist[0, 0]:+.6f} (distorsion radiale)")
    print(f"   k2 = {dist[0, 1]:+.6f} (distorsion radiale)")
    print(f"   p1 = {dist[0, 2]:+.6f} (distorsion tangentielle)")
    print(f"   p2 = {dist[0, 3]:+.6f} (distorsion tangentielle)")
    print(f"   k3 = {dist[0, 4]:+.6f} (distorsion radiale)")
    
    print(f"\n✓ Fichiers mis à jour:")
    print(f"   📄 {camera_matrix_path}")
    print(f"   📄 {distortion_path}")
    
    # Évaluer la qualité de la calibration
    print("\n" + "="*70)
    print("ÉVALUATION DE LA QUALITÉ")
    print("="*70)
    
    if ret < 0.5:
        quality = "EXCELLENTE ✨"
    elif ret < 1.0:
        quality = "BONNE ✓"
    elif ret < 2.0:
        quality = "ACCEPTABLE ⚠️"
    else:
        quality = "FAIBLE ❌"
    
    print(f"\nQualité de la calibration: {quality}")
    print(f"Erreur RMS: {ret:.4f} pixels")
    
    if ret < 1.0:
        print("\n👍 Excellente calibration! Les paramètres peuvent être utilisés en production.")
    elif ret < 2.0:
        print("\n⚠️  Calibration acceptable, mais pourrait être améliorée.")
        print("   Suggestions:")
        print("   - Ajouter plus d'images avec différents angles")
        print("   - Assurer un meilleur éclairage")
        print("   - Vérifier que le damier est complètement visible")
    else:
        print("\n❌ Calibration de qualité insuffisante.")
        print("   Recommandations:")
        print("   - Recapturer les images avec plus de soin")
        print("   - Utiliser un éclairage uniforme")
        print("   - Assurer que le damier couvre différentes parties de l'image")
        print("   - Vérifier que le damier est plat et non déformé")
    
    print("\n" + "="*70)
    print("✓ RECALIBRATION TERMINÉE")
    print("="*70)

def main():
    """
    Fonction principale pour recalibrer à partir d'images existantes.
    """
    print("\n" + "="*70)
    print("PROGRAMME DE RECALIBRATION DE CAMÉRA")
    print("="*70)
    print(f"\nCe programme va:")
    print(f"1. Charger les images du dossier '{INPUT_DIRECTORY}'")
    print(f"2. Détecter les damiers dans les images")
    print(f"3. Calculer les paramètres de calibration")
    print(f"4. Mettre à jour les fichiers:")
    print(f"   - camera_matrix.txt")
    print(f"   - distortion_coefficients.txt")
    print("="*70)
    
    recalibrate_from_existing_images()

if __name__ == "__main__":
    main()
