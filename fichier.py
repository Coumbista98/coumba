from flask import Flask, request, render_template, send_file, flash, redirect, url_for
import pandas as pd
import numpy as np
import os
import uuid
from werkzeug.utils import secure_filename

# Initialisation de l'application Flask
app = Flask(__name__)
app.secret_key = 'votre_cle_secrete_ici'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Création du dossier uploads s'il n'existe pas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Vérifie si l'extension du fichier est autorisée"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_dataframe(df):
    """Nettoyage de base du dataframe"""
    cleaned_df = df.copy()
    
    # Conversion des colonnes numériques
    for col in cleaned_df.columns:
        try:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='raise')
            continue
        except:
            pass
        
        # Traitement des valeurs manquantes pour les colonnes non numériques
        if cleaned_df[col].dtype == object:
            cleaned_df[col] = cleaned_df[col].replace(
                ['n/a', 'NA', 'na', 'NaN', 'NULL', 'null', '--', '?', ''], np.nan)
            
            unique_vals = cleaned_df[col].dropna().unique()
            if all(str(x).upper() in ['Y', 'N', 'OUI', 'NON', 'YES', 'NO'] for x in unique_vals if pd.notnull(x)):
                cleaned_df[col] = cleaned_df[col].apply(
                    lambda x: str(x).upper() if pd.notnull(x) else np.nan)
    
    return cleaned_df

def handle_missing_values(df):
    """Gestion des valeurs manquantes"""
    missing_before = df.isnull().sum().to_dict()
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        else:
            if len(df[col].mode()) > 0:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
            else:
                df[col] = df[col].fillna('')
    
    missing_after = df.isnull().sum().to_dict()
    
    report = {
        'before': missing_before,
        'after': missing_after,
        'message': "Valeurs manquantes traitées"
    }
    return df, report

def handle_outliers(df):
    """Détection et traitement des valeurs aberrantes"""
    report = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        
        if iqr > 0:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                df[col] = np.where(df[col] < lower_bound, lower_bound,
                                  np.where(df[col] > upper_bound, upper_bound, df[col]))
                
                report[col] = {
                    'count': outlier_count,
                    'lower_bound': round(lower_bound, 2),
                    'upper_bound': round(upper_bound, 2)
                }
    
    if not report:
        report = {"message": "Aucune valeur aberrante détectée"}
    return df, report

def handle_duplicates(df):
    """Gestion des doublons"""
    duplicates = df[df.duplicated()]
    dup_count = len(duplicates)
    
    if dup_count > 0:
        df = df.drop_duplicates()
        report = {
            'count': dup_count,
            'message': f"{dup_count} doublons supprimés"
        }
        return df, report
    
    return df, {"message": "Aucun doublon détecté"}

# Routes de l'application
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/uploads')
def uploads():
    return render_template('uploads.html')
@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/traitement', methods=['POST'])
def traitement():
    if 'file' not in request.files:
        flash('Aucun fichier sélectionné', 'error')
        return redirect(url_for('upload'))
    
    file = request.files['file']
    if file.filename == '':
        flash('Aucun fichier sélectionné', 'error')
        return redirect(url_for('upload'))
    
    if not allowed_file(file.filename):
        flash('Seuls les fichiers CSV sont acceptés', 'error')
        return redirect(url_for('upload'))
    
    try:
        # Sauvegarde et lecture du fichier
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        df = pd.read_csv(filepath)
        if df.empty:
            flash('Le fichier CSV est vide', 'error')
            return redirect(url_for('upload'))
        
        original_shape = df.shape
        
        # Traitement des données
        df = clean_dataframe(df)
        df, missing_report = handle_missing_values(df) # valeurs manquantes
        df, outliers_report = handle_outliers(df) # valeurs aberantes
        df, duplicates_report = handle_duplicates(df) # valeurs dupliquées
        
        # Sauvegarde du résultat
        output_filename = f"cleaned_{uuid.uuid4().hex}.csv"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        df.to_csv(output_path, index=False)
        
        # Préparation des résultats pour le template
        final_shape = df.shape
        sample_data = df.head(10).to_dict('records')
        column_types = {col: str(df[col].dtype) for col in df.columns}
        
        return render_template('result.html',
                            original_shape=original_shape,
                            final_shape=final_shape,
                            missing_report=missing_report,
                            outliers_report=outliers_report,
                            duplicates_report=duplicates_report,
                            sample_data=sample_data,
                            column_types=column_types,
                            download_file=output_filename)
    
    except pd.errors.EmptyDataError:
        flash('Le fichier CSV est vide ou corrompu', 'error')
        return redirect(url_for('upload'))
    except Exception as e:
        flash(f"Erreur de traitement: {str(e)}", 'error')
        return redirect(url_for('upload'))

@app.route('/download')
def download():
    return send_file('static/traiter.csv', 
                    as_attachment=True, 
                    mimetype='text/csv',
                    download_name='traiter.csv')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render définit automatiquement la variable d'env PORT
    app.run(host='0.0.0.0', port=port, debug=True)