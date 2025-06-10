import cv2
import numpy as np
from ultralytics import YOLO
import os

class SecuritySystem:
    def __init__(self):
        # Configuración inicial
        self.winName = 'Sistema de Seguridad IA'
        cv2.namedWindow(self.winName, cv2.WINDOW_AUTOSIZE)
        
        # Cargar modelos
        self.load_models()
        
        # Configurar cámara
        self.cap = self.setup_camera()
        
        # Estados
        self.status = "Todo en orden"
        self.status_color = (0, 255, 0)  # Verde
        
    def load_models(self):
        """Carga los modelos de detección"""
        # Cargar nombres de clases COCO
        self.classNames = []
        classFile = 'coco.names'
        if os.path.exists(classFile):
            with open(classFile, 'rt') as f:
                self.classNames = f.read().rstrip('\n').split('\n')
        else:
            print(f"Advertencia: No se encontró el archivo {classFile}")
            self.classNames = [f"Class_{i}" for i in range(90)]  # Placeholder
            
        # Cargar MobileNet-SSD (detección rápida)
        configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightsPath = 'frozen_inference_graph.pb'
        
        if os.path.exists(configPath) and os.path.exists(weightsPath):
            self.net = cv2.dnn_DetectionModel(weightsPath, configPath)
            self.net.setInputSize(320, 320)
            self.net.setInputScale(1.0 / 127.5)
            self.net.setInputMean((127.5, 127.5, 127.5))
            self.net.setInputSwapRB(True)
        else:
            print("Advertencia: No se encontraron los archivos del modelo MobileNet-SSD")
            self.net = None
            
        # Cargar YOLOv8 (detección avanzada)
        try:
            self.yolo_model = YOLO('yolov8m.pt')
        except Exception as e:
            print(f"Error al cargar YOLOv8: {e}")
            self.yolo_model = None
    
    def setup_camera(self):
        """Configura la captura de video"""
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            exit()
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap
    
    def detect_events(self, frame):
        """Detección de eventos importantes usando YOLOv8"""
        if self.yolo_model is None:
            return None
            
        results = self.yolo_model(frame, verbose=False)[0]
        personas = []
        armas = []

        for box in results.boxes:
            cls_id = int(box.cls)
            label = self.yolo_model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == "person":
                personas.append((x1, y1, x2, y2))
            elif label in ["knife", "sports ball", "scissors", "gun"]:  # Objetos peligrosos
                armas.append((x1, y1, x2, y2))

        # Verificar proximidad persona + arma
        for px1, py1, px2, py2 in personas:
            for ax1, ay1, ax2, ay2 in armas:
                # Calcular distancia entre centros
                persona_center = ((px1 + px2) // 2, (py1 + py2) // 2)
                arma_center = ((ax1 + ax2) // 2, (ay1 + ay2) // 2)
                distancia = np.linalg.norm(np.array(persona_center) - np.array(arma_center))
                
                if distancia < 150:  # Aumentado el umbral de distancia
                    return "robo"

        return None
    
    def detect_covered_camera(self, frame):
        """Detección si la cámara está tapada"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = cv2.mean(gray)[0]
        
        # Umbral adaptativo basado en la desviación estándar
        std_dev = np.std(gray)
        threshold = max(20, 50 - std_dev)  # Ajuste dinámico
        
        return avg_brightness < threshold
    
    def run_detection(self):
        """Bucle principal de detección"""
        print("Sistema de seguridad activado. Presiona 'q' para salir")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: No se pudo capturar el frame")
                    break

                # Detección rápida con MobileNet-SSD (si está disponible)
                if self.net is not None:
                    classIds, confs, bbox = self.net.detect(frame, confThreshold=0.5)
                    if len(classIds) != 0:
                        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                            cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                            cv2.putText(frame, f"{self.classNames[classId - 1]} {int(confidence * 100)}%",
                                        (box[0] + 10, box[1] + 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Detección de eventos importantes
                evento = self.detect_events(frame)

                # Actualizar estado del sistema
                self.status = "Todo en orden"
                self.status_color = (0, 255, 0)  # Verde

                if self.detect_covered_camera(frame):
                    evento = "cámara_tapada"
                    self.status = "❌ Cámara tapada"
                    self.status_color = (0, 0, 255)
                elif evento == "robo":
                    self.status = "🚨 ¡Alerta de robo!"
                    self.status_color = (0, 0, 255)

                # Mostrar estado en pantalla
                cv2.putText(frame, self.status, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.status_color, 2)
                cv2.imshow(self.winName, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("Sistema detenido correctamente")

if __name__ == "__main__":
    system = SecuritySystem()
    system.run_detection()