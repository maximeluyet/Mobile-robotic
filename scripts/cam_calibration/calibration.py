import numpy as np
import cv2
import os
import pickle
from datetime import datetime

# Configuration
CAMERA_ID = 0  # ID de la caméra (0 pour webcam intégrée, 1 pour caméra externe)
IMAGE_RES = (1920, 1080)  # Résolution de la caméra
NUM_IMAGES_REQUIRED = 15  # Nombre d'images requises pour la calibration
CHESSBOARD_SIZE = (9, 6)  # Nombre de coins intérieurs du damier (colonnes, lignes)
SQUARE_SIZE = 2.5  # Taille d'un carré en cm (pour mesures réelles)

# Répertoire de sortie - toujours dans le même dossier que ce script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIRECTORY = os.path.join(SCRIPT_DIR, 'calibration_output')

# Liste de tailles alternatives à essayer si la détection échoue
ALTERNATIVE_SIZES = [(9, 6)]

def capture_calibration_images():
    """
    Capture interactive d'images de calibration avec l'utilisateur.
    L'utilisateur doit prendre 15
      photos d'un damier sous différents angles.
    
    Returns:
        List of captured images
        Best detected chessboard size
    """
    print("="*70)
    print("CALIBRATION DE CAMÉRA - CAPTURE D'IMAGES")
    print("="*70)
    print(f"\nVous devez prendre {NUM_IMAGES_REQUIRED} photos d'un damier (chessboard)")
    print("\nInstructions:")
    print("- Placez le damier sous différents angles et distances")
    print("- Assurez-vous que tout le damier est visible dans chaque photo")
    print("- Utilisez un bon éclairage")
    print("- Appuyez sur 'C' ou ESPACE pour capturer une image")
    print("- Appuyez sur 'Q' ou ESC pour quitter")
    print(f"\nConfiguration par défaut: Damier {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} coins intérieurs")
    print("="*70)
    
    input("\nAppuyez sur ENTRÉE pour démarrer la capture...")
    
    # Ouvrir la caméra
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_RES[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_RES[1])
    
    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la caméra {CAMERA_ID}")
        return [], None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n Caméra ouverte - Résolution: {width}x{height}")
    
    # Créer le répertoire de sortie
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    
    captured_images = []
    detected_size = None
    img_counter = 0
    
    print("\n Capture en cours...")
    print(f"Objectif: {NUM_IMAGES_REQUIRED} images avec damier détecté\n")
    
    while img_counter < NUM_IMAGES_REQUIRED:
        ret, frame = cap.read()
        
        if not ret:
            print("Erreur: Impossible de capturer l'image")
            break
        
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Essayer de détecter le damier
        chessboard_detected = False
        current_size = CHESSBOARD_SIZE
        
        # Essayer d'abord avec la taille par défaut
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        ret_chess, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, flags)
        
        if ret_chess:
            chessboard_detected = True
            detected_size = CHESSBOARD_SIZE
        else:
            # Essayer d'autres tailles si première détection échoue
            for size in ALTERNATIVE_SIZES:
                if size == CHESSBOARD_SIZE:
                    continue
                ret_chess, corners = cv2.findChessboardCorners(gray, size, flags)
                if ret_chess:
                    chessboard_detected = True
                    current_size = size
                    if detected_size is None:
                        detected_size = size
                    break
        
        # Afficher les coins si détectés
        if chessboard_detected:
            cv2.drawChessboardCorners(display_frame, current_size, corners, ret_chess)
            cv2.putText(display_frame, f"DAMIER DETECTE! ({current_size[0]}x{current_size[1]})", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, "Appuyez sur 'C' ou ESPACE pour capturer", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Damier non detecte", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(display_frame, "Positionnez le damier dans le cadre", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Afficher le compteur
        progress_text = f"Images capturees: {img_counter}/{NUM_IMAGES_REQUIRED}"
        cv2.putText(display_frame, progress_text, 
                   (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Barre de progression
        bar_width = int((img_counter / NUM_IMAGES_REQUIRED) * (width - 40))
        cv2.rectangle(display_frame, (20, height - 60), (width - 20, height - 40), (100, 100, 100), 2)
        if bar_width > 0:
            cv2.rectangle(display_frame, (20, height - 60), (20 + bar_width, height - 40), (0, 255, 0), -1)
        
        cv2.imshow('Calibration de Camera', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Quitter
        if key == ord('q') or key == 27:
            print("\n⚠ Capture annulée par l'utilisateur")
            break
        
        # Capturer l'image
        elif key == ord('c') or key == 32:  # 'c' ou espace
            if chessboard_detected:
                # Sauvegarder l'image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                img_name = os.path.join(OUTPUT_DIRECTORY, f"calib_{img_counter:02d}_{timestamp}.jpg")
                cv2.imwrite(img_name, frame)
                captured_images.append(frame.copy())
                
                print(f"✓ [{img_counter + 1}/{NUM_IMAGES_REQUIRED}] Image capturée: {os.path.basename(img_name)}")
                img_counter += 1
                
                # Petit feedback visuel
                cv2.putText(display_frame, "CAPTURE!", (width//2 - 100, height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.imshow('Calibration de Camera', display_frame)
                cv2.waitKey(200)  # Pause de 200ms
            else:
                print("⚠ Impossible de capturer - Damier non détecté!")
                cv2.putText(display_frame, "ECHEC - Damier non detecte!", 
                           (width//2 - 200, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Calibration de Camera', display_frame)
                cv2.waitKey(500)
    
    cap.release()
    cv2.destroyAllWindows()
    
    if img_counter == NUM_IMAGES_REQUIRED:
        print(f"\n✓ Capture terminée! {NUM_IMAGES_REQUIRED} images acquises")
    
    return captured_images, detected_size

def calibrate_camera(images, chessboard_size):
    """
    Calibre la caméra en utilisant les images capturées.
    Calcule les paramètres intrinsèques de la caméra.
    
    Args:
        images: Liste des images capturées
        chessboard_size: Taille du damier détectée
    
    Returns:
        ret: Erreur de reprojection RMS
        mtx: Matrice de la caméra (paramètres intrinsèques)
        dist: Coefficients de distorsion
        rvecs: Vecteurs de rotation
        tvecs: Vecteurs de translation
    """
    print("\n" + "="*70)
    print("CALIBRATION DE LA CAMÉRA")
    print("="*70)
    
    if not images:
        print("Aucune image disponible pour la calibration")
        return None, None, None, None, None
    
    # Utiliser la taille détectée ou celle par défaut
    if chessboard_size is None:
        chessboard_size = CHESSBOARD_SIZE
    
    print(f"\nTaille du damier utilisée: {chessboard_size}")
    print(f"Nombre d'images: {len(images)}")
    
    # Préparer les points 3D du damier (dans le repère du damier)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE  # Mise à l'échelle avec la taille réelle des carrés
    
    # Tableaux pour stocker les points
    objpoints = []  # Points 3D dans l'espace réel
    imgpoints = []  # Points 2D dans le plan image
    
    print("\nTraitement des images...")
    
    for idx, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Détecter les coins du damier
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, flags)
        
        if ret:
            objpoints.append(objp)
            
            # Raffiner la position des coins
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_refined)
            
            print(f"✓ Image {idx + 1}/{len(images)} - Damier détecté et traité")
        else:
            print(f"⚠ Image {idx + 1}/{len(images)} - Damier non détecté (ignorée)")
    
    if len(objpoints) < 3:
        print(f"\n Erreur: Pas assez d'images valides ({len(objpoints)} < 3)")
        print("La calibration nécessite au moins 3 images avec damier détecté")
        return None, None, None, None, None
    
    print(f"\n✓ {len(objpoints)} images valides pour la calibration")
    print("\nCalcul des paramètres intrinsèques de la caméra...")
    
    # Calibration de la caméra
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    # Sauvegarder les résultats
    print("\nSauvegarde des résultats...")
    
    calibration_data = {
        'camera_matrix': mtx,
        'distortion_coefficients': dist,
        'rotation_vectors': rvecs,
        'translation_vectors': tvecs,
        'reprojection_error': ret,
        'chessboard_size': chessboard_size,
        'num_images_used': len(objpoints),
        'image_size': gray.shape[::-1]
    }
    
    # Sauvegarder en pickle
    with open(os.path.join(OUTPUT_DIRECTORY, 'calibration_data.pkl'), 'wb') as f:
        pickle.dump(calibration_data, f)
    
    # Sauvegarder en fichiers texte
    np.savetxt(os.path.join(OUTPUT_DIRECTORY, 'camera_matrix.txt'), mtx, 
               header='Camera Matrix (Intrinsic Parameters)', fmt='%.6f')
    np.savetxt(os.path.join(OUTPUT_DIRECTORY, 'distortion_coefficients.txt'), dist, 
               header='Distortion Coefficients [k1, k2, p1, p2, k3]', fmt='%.6f')
    
    # Afficher les résultats
    print("\n" + "="*70)
    print("RÉSULTATS DE LA CALIBRATION")
    print("="*70)
    print(f"\n Erreur de reprojection RMS: {ret:.4f} pixels")
    print(f" Images utilisées: {len(objpoints)}/{len(images)}")
    print(f"\nMatrice de la caméra (paramètres intrinsèques):")
    print(f"  fx = {mtx[0, 0]:.2f}  (distance focale en x)")
    print(f"  fy = {mtx[1, 1]:.2f}  (distance focale en y)")
    print(f"  cx = {mtx[0, 2]:.2f}  (centre optique en x)")
    print(f"  cy = {mtx[1, 2]:.2f}  (centre optique en y)")
    print(f"\nCoefficients de distorsion:")
    print(f"  k1 = {dist[0, 0]:.6f}")
    print(f"  k2 = {dist[0, 1]:.6f}")
    print(f"  p1 = {dist[0, 2]:.6f}")
    print(f"  p2 = {dist[0, 3]:.6f}")
    print(f"  k3 = {dist[0, 4]:.6f}")
    
    print(f"\n✓ Résultats sauvegardés dans: {OUTPUT_DIRECTORY}/")
    print("  - calibration_data.pkl (données complètes)")
    print("  - camera_matrix.txt")
    print("  - distortion_coefficients.txt")
    
    return ret, mtx, dist, rvecs, tvecs

def demonstrate_undistortion(images, mtx, dist):
    """
    Démontre la correction de distorsion sur quelques images.
    
    Args:
        images: Liste des images capturées
        mtx: Matrice de la caméra
        dist: Coefficients de distorsion
    """
    if not images or mtx is None:
        return
    
    print("\n" + "="*70)
    print("DÉMONSTRATION DE LA CORRECTION DE DISTORSION")
    print("="*70)
    
    undistorted_dir = os.path.join(OUTPUT_DIRECTORY, 'undistorted_samples')
    if not os.path.exists(undistorted_dir):
        os.makedirs(undistorted_dir)
    
    # Corriger quelques images d'exemple
    num_samples = min(3, len(images))
    print(f"\nCorrection de {num_samples} images d'exemple...")
    
    for i in range(num_samples):
        img = images[i]
        h, w = img.shape[:2]
        
        # Obtenir la matrice de caméra optimale
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        
        # Corriger la distorsion
        undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)
        
        # Rogner l'image selon la ROI
        x, y, w_roi, h_roi = roi
        undistorted_cropped = undistorted[y:y+h_roi, x:x+w_roi]
        
        # Sauvegarder
        output_path = os.path.join(undistorted_dir, f'undistorted_sample_{i+1}.jpg')
        cv2.imwrite(output_path, undistorted_cropped)
        
        print(f"✓ Image {i+1} corrigée et sauvegardée")
    
    print(f"\n✓ Images corrigées sauvegardées dans: {undistorted_dir}/")

def main():
    """
    Fonction principale pour exécuter la calibration de caméra interactive.
    """
    print("\n" + "="*70)
    print("PROGRAMME DE CALIBRATION DE CAMÉRA")
    print("="*70)
    print("\nCe programme va:")
    print(f"1. Capturer {NUM_IMAGES_REQUIRED} images d'un damier")
    print("2. Calculer les paramètres intrinsèques de la caméra")
    print("3. Sauvegarder les résultats de calibration")
    print("="*70)
    
    # Étape 1: Capture des images
    captured_images, detected_size = capture_calibration_images()
    
    if len(captured_images) < 3:
        print(f"\nÉchec: Pas assez d'images capturées ({len(captured_images)} < 3)")
        print("La calibration nécessite au moins 3 images")
        return
    
    # Étape 2: Calibration
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(captured_images, detected_size)
    
    if mtx is None:
        print("\nLa calibration a échoué")
        return
    
    # Étape 3: Démonstration (optionnel)
    demonstrate_undistortion(captured_images, mtx, dist)
    
    print("\n" + "="*70)
    print("✓ CALIBRATION TERMINÉE AVEC SUCCÈS!")
    print("="*70)
    print("\nVous pouvez maintenant utiliser les paramètres de calibration")
    print("pour corriger la distorsion dans vos applications.")
    print("="*70)

if __name__ == "__main__":
    main()

