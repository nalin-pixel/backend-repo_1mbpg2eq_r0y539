from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import io
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
import optuna
import shap
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import base64

app = FastAPI(title="CMMD Hybrid Breast Cancer Detection API")


class TrainResponse(BaseModel):
    accuracy: float
    auc: float
    note: str


# Simple CNN backbone for demo purposes
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def preprocess_image(img: Image.Image, size: int = 224) -> torch.Tensor:
    gray = img.convert("L")
    tfm = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    return tfm(gray)


def encode_image(img_np: np.ndarray) -> str:
    _, buf = cv2.imencode('.png', img_np)
    return base64.b64encode(buf.tobytes()).decode('utf-8')


@app.post("/train", response_model=TrainResponse)
async def train_endpoint(
    clinical_csv: UploadFile = File(..., description="CMMD clinical CSV file"),
    images: List[UploadFile] = File(..., description="Mammogram images JPG/PNG"),
    image_map: str = Form(..., description="CSV mapping: filename,label (0/1), optional patient_id"),
    epochs: int = Form(3),
    image_size: int = Form(224)
):
    try:
        # Load clinical data
        clinical_bytes = await clinical_csv.read()
        clinical_df = pd.read_csv(io.BytesIO(clinical_bytes))
        # Basic preprocessing: drop non-numeric except label/ID, impute, scale
        label_col = None
        for cand in ["label", "target", "malignant", "cancer", "y"]:
            if cand in clinical_df.columns:
                label_col = cand
                break
        if label_col is None:
            return JSONResponse(status_code=400, content={"error": "Clinical CSV must contain a label column (label/target/malignant/cancer/y)"})

        y_tab = clinical_df[label_col].astype(int).values
        X_tab = clinical_df.drop(columns=[label_col])
        # Convert categoricals
        for col in X_tab.columns:
            if X_tab[col].dtype == 'object':
                X_tab[col] = X_tab[col].astype('category')
        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(X_tab, y_tab, test_size=0.2, random_state=42, stratify=y_tab)

        # LightGBM with Optuna tuning
        def objective(trial: optuna.Trial):
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 16, 256, log=True),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            }
            dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
            dval = lgb.Dataset(X_val, label=y_val, free_raw_data=False)
            model = lgb.train(params, dtrain, valid_sets=[dval], num_boost_round=500, early_stopping_rounds=50, verbose_eval=False)
            preds = model.predict(X_val)
            return roc_auc_score(y_val, preds)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)
        best_params = study.best_params
        best_params.update({'objective': 'binary', 'metric': 'auc', 'verbosity': -1})
        dtrain_full = lgb.Dataset(X_tab, label=y_tab, free_raw_data=False)
        lgb_model = lgb.train(best_params, dtrain_full, num_boost_round=study.best_trial.user_attrs.get('best_iteration', 200))

        # SHAP explainability for tabular
        explainer = shap.TreeExplainer(lgb_model)
        shap_values = explainer.shap_values(X_val)
        # Aggregate feature importance
        shap_abs_mean = np.mean(np.abs(shap_values), axis=0) if isinstance(shap_values, np.ndarray) else np.mean(np.abs(shap_values[1]), axis=0)
        top_features = list(pd.Series(shap_abs_mean, index=X_val.columns).sort_values(ascending=False).head(10).index)

        # Load image mapping
        map_df = pd.read_csv(io.StringIO(image_map))
        if not {"filename", "label"}.issubset(set(map_df.columns)):
            return JSONResponse(status_code=400, content={"error": "image_map must have columns: filename,label"})
        img_label_map = {row['filename']: int(row['label']) for _, row in map_df.iterrows()}

        # Build tensors
        imgs = []
        y_img = []
        name_set = set([f.filename for f in images])
        for up in images:
            if up.filename not in img_label_map:
                continue
            data = await up.read()
            img = Image.open(io.BytesIO(data)).convert('L')
            tensor = preprocess_image(img, size=image_size)
            imgs.append(tensor)
            y_img.append(img_label_map[up.filename])
        if len(imgs) < 4:
            note = "Not enough labeled images provided; image model will be trained minimally."
        else:
            note = ""
        if len(imgs) == 0:
            # No image training possible
            cnn_model = SimpleCNN()
            img_auc = 0.5
            img_acc = 0.5
            img_probs = np.zeros(len(X_val)) + 0.5
        else:
            X_img = torch.stack(imgs)
            y_img_t = torch.tensor(y_img, dtype=torch.long)
            # Split
            idx = np.arange(len(y_img))
            train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_img)
            Ximg_train = X_img[train_idx]
            Ximg_val = X_img[val_idx]
            yimg_train = y_img_t[train_idx]
            yimg_val = y_img_t[val_idx]

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            cnn_model = SimpleCNN().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-3)

            for ep in range(epochs):
                cnn_model.train()
                optimizer.zero_grad()
                out = cnn_model(Ximg_train.to(device))
                loss = criterion(out, yimg_train.to(device))
                loss.backward()
                optimizer.step()

            cnn_model.eval()
            with torch.no_grad():
                logits = cnn_model(Ximg_val.to(device))
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = (probs > 0.5).astype(int)
                img_acc = accuracy_score(yimg_val.cpu().numpy(), preds)
                if len(np.unique(yimg_val.cpu().numpy())) == 1:
                    img_auc = 0.5
                else:
                    img_auc = roc_auc_score(yimg_val.cpu().numpy(), probs)

            # Generate simple Grad-CAM for the first validation image
            try:
                target_layer = cnn_model.features[-2]  # AdaptiveAvgPool2d is last; take previous conv
                cam = GradCAM(model=cnn_model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
                targets = [ClassifierOutputTarget(1)]
                input_img = Ximg_val[0:1].to(device)
                grayscale_cam = cam(input_tensor=input_img, targets=targets)[0]
                rgb_img = Ximg_val[0].permute(1, 2, 0).cpu().numpy()
                rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
                rgb_img = np.repeat(rgb_img, 3, axis=2)
                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                cam_b64 = encode_image(cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
            except Exception:
                cam_b64 = ""

            # Image probabilities aligned to tabular val size (broadcast if needed)
            img_probs = probs

        # Tabular validation predictions
        tab_val_preds = lgb_model.predict(X_val)
        tab_pred_labels = (tab_val_preds > 0.5).astype(int)
        tab_acc = accuracy_score(y_val, tab_pred_labels)
        tab_auc = roc_auc_score(y_val, tab_val_preds) if len(np.unique(y_val)) > 1 else 0.5

        # Late fusion: average probabilities (handle size mismatch by truncation/tiling)
        def align(arr, n):
            if len(arr) == 0:
                return np.zeros(n) + 0.5
            if len(arr) == n:
                return arr
            if len(arr) > n:
                return arr[:n]
            reps = int(np.ceil(n / len(arr)))
            return np.tile(arr, reps)[:n]

        fused_probs = (tab_val_preds + align(img_probs, len(tab_val_preds))) / 2.0
        fused_preds = (fused_probs > 0.5).astype(int)
        acc = accuracy_score(y_val, fused_preds)
        auc = roc_auc_score(y_val, fused_probs) if len(np.unique(y_val)) > 1 else 0.5

        note_msg = note or ""
        return TrainResponse(accuracy=round(float(acc), 4), auc=round(float(auc), 4), note=note_msg)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


class InferenceResponse(BaseModel):
    probability: float
    image_gradcam_png_b64: Optional[str] = None
    shap_top_features: Optional[List[str]] = None


@app.post("/infer", response_model=InferenceResponse)
async def infer_endpoint(
    clinical_csv: UploadFile = File(...),
    image: UploadFile = File(...),
):
    # Minimal demo: fit simple models on provided single-row CSV duplicated for scaler, run one forward on image
    try:
        df = pd.read_csv(io.BytesIO(await clinical_csv.read()))
        if df.shape[0] == 0:
            return JSONResponse(status_code=400, content={"error": "Empty clinical CSV"})
        # Use heuristic label if provided else default 0
        y = df[df.columns[-1]].astype(int).values if df.shape[1] > 1 else np.zeros(len(df))
        X = df.drop(columns=[df.columns[-1]]) if df.shape[1] > 1 else df
        for c in X.columns:
            if X[c].dtype == 'object':
                X[c] = X[c].astype('category')
        model = lgb.LGBMClassifier(objective='binary')
        model.fit(X, y)
        tab_prob = float(model.predict_proba(X.iloc[[0]])[0, 1])

        # Image CNN quick pass
        data = await image.read()
        img = Image.open(io.BytesIO(data)).convert('L')
        tensor = preprocess_image(img).unsqueeze(0)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cnn = SimpleCNN().to(device)
        with torch.no_grad():
            prob = torch.softmax(cnn(tensor.to(device)), dim=1)[0, 1].item()

        fused = (tab_prob + prob) / 2.0

        # Grad-CAM heatmap (untrained model; illustrative)
        try:
            target_layer = cnn.features[-2]
            cam = GradCAM(model=cnn, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
            targets = [ClassifierOutputTarget(1)]
            grayscale_cam = cam(input_tensor=tensor.to(device), targets=targets)[0]
            rgb_img = tensor[0].permute(1, 2, 0).cpu().numpy()
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
            rgb_img = np.repeat(rgb_img, 3, axis=2)
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cam_b64 = encode_image(cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
        except Exception:
            cam_b64 = None

        # SHAP top features from the trained small model
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X)
            sv_vals = sv if isinstance(sv, np.ndarray) else sv[1]
            top_feats = list(pd.Series(np.mean(np.abs(sv_vals), axis=0), index=X.columns).sort_values(ascending=False).head(5).index)
        except Exception:
            top_feats = None

        return InferenceResponse(probability=float(fused), image_gradcam_png_b64=cam_b64, shap_top_features=top_feats)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/test")
async def test():
    return {"status": "ok"}
