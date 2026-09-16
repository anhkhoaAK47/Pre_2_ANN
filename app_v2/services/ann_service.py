"""
ANN Service Module for Disease Prediction

This module provides interface to the trained PyTorch ANN model for
disease classification with 577 diseases and 377 symptoms.

Author: AI Medical Chatbot Team
Date: 2025-10-18
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

from .active_questioning import (
    Hypothesis,
    ActiveQuestioningEngine,
    Question,
    QuestioningSession,
    create_disease_symptom_matrix
)
from .department_questioning import (
    DepartmentBasedQuestioningEngine,
    DepartmentHypothesis,
    DepartmentQuestion,
    DepartmentQuestioningSession,
    QuestioningStrategy
)
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result from disease prediction"""
    top_disease: str
    top_confidence: float
    top_k_predictions: List[Tuple[str, float]]
    all_probabilities: np.ndarray
    confidence_category: str  # HIGH, MEDIUM, LOW
    needs_questioning: bool
    hypotheses: List[Hypothesis]


class ANNDiseaseClassifier(nn.Module):
    """
    PyTorch ANN model for disease classification.
    Architecture: 377 → 512 → 256 → 128 → 577
    """

    def __init__(self, input_dim: int, num_classes: int, dropout_rates=(0.3, 0.3, 0.2)):
        super(ANNDiseaseClassifier, self).__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rates[0])

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rates[1])

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout_rates[2])

        self.fc4 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        return x


class ANNService:
    """
    Service for ANN model prediction and active questioning integration.
    """

    def __init__(
        self,
        model_path: str,
        label_encoder_path: str,
        train_data_path: str,
        confusion_matrix_path: Optional[str] = None,
        department_map: Optional[Dict[str, str]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize ANN service.

        Args:
            model_path: Path to best_model.pth
            label_encoder_path: Path to label_encoder.pkl
            train_data_path: Path to train.csv (for symptom names and disease-symptom matrix)
            confusion_matrix_path: Path to confusion matrix data (optional)
            department_map: Dict mapping disease_name -> department_name (for department-based questioning)
            device: PyTorch device (auto-detected if None)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load label encoder
        logger.info(f"Loading label encoder from {label_encoder_path}")
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

        self.num_classes = len(self.label_encoder.classes_)
        logger.info(f"Loaded label encoder: {self.num_classes} diseases")

        # Load training data to get symptom names
        logger.info(f"Loading training data from {train_data_path}")
        self.train_df = pd.read_csv(train_data_path)
        self.symptom_columns = [col for col in self.train_df.columns if col != 'disease']
        self.num_symptoms = len(self.symptom_columns)
        logger.info(f"Loaded training data: {len(self.train_df)} samples, {self.num_symptoms} symptoms")

        # Create symptom name to index mapping
        self.symptom_to_idx = {symptom: idx for idx, symptom in enumerate(self.symptom_columns)}
        self.idx_to_symptom = {idx: symptom for symptom, idx in self.symptom_to_idx.items()}

        # Load model
        logger.info(f"Loading ANN model from {model_path}")
        self.model = ANNDiseaseClassifier(self.num_symptoms, self.num_classes)
        self.model = self.model.to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"Model loaded successfully")
        logger.info(f"Best epoch: {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"Validation accuracy: {checkpoint.get('val_acc', 0)*100:.2f}%")

        # Create disease-symptom association matrix
        logger.info("Creating disease-symptom association matrix...")
        self.disease_symptom_matrix, _ = create_disease_symptom_matrix(
            self.train_df, self.label_encoder
        )

        # Load confusion matrix data if provided
        self.confusion_matrix_data = None
        if confusion_matrix_path and Path(confusion_matrix_path).exists():
            logger.info(f"Loading confusion matrix from {confusion_matrix_path}")
            self.confusion_matrix_data = pd.read_csv(confusion_matrix_path)

        # Initialize disease-based questioning engine
        self.questioning_engine = ActiveQuestioningEngine(
            symptom_columns=self.symptom_columns,
            confusion_matrix_data=self.confusion_matrix_data,
            max_questions=10,
            confidence_threshold=0.85,
            gap_threshold=0.40
        )

        # Initialize department-based questioning engine (if department map provided)
        self.dept_questioning_engine = None
        if department_map:
            logger.info("Initializing department-based questioning engine...")
            disease_names = list(self.label_encoder.classes_)
            self.dept_questioning_engine = DepartmentBasedQuestioningEngine(
                disease_symptom_matrix=self.disease_symptom_matrix,
                symptom_names=self.symptom_columns,
                disease_names=disease_names,
                department_map=department_map,
                strategy=QuestioningStrategy.HYBRID,
                max_questions=10,
                confidence_threshold=0.85,
                gap_threshold=0.40
            )
            logger.info("✅ Department-based questioning engine initialized")

        # Store active sessions
        self.active_sessions: Dict[str, QuestioningSession] = {}
        self.dept_active_sessions: Dict[str, DepartmentQuestioningSession] = {}

        logger.info("ANNService initialization complete")

    def create_symptom_vector(
        self,
        symptom_names: List[str],
        warn_unknown: bool = True
    ) -> np.ndarray:
        """
        Create binary symptom vector from symptom names.

        Args:
            symptom_names: List of symptom names (Vietnamese)
            warn_unknown: Whether to log warning for unknown symptoms

        Returns:
            Binary numpy array of shape (num_symptoms,)
        """
        vector = np.zeros(self.num_symptoms, dtype=np.float32)

        unknown_symptoms = []
        for symptom in symptom_names:
            if symptom in self.symptom_to_idx:
                idx = self.symptom_to_idx[symptom]
                vector[idx] = 1.0
            else:
                unknown_symptoms.append(symptom)

        if warn_unknown and unknown_symptoms:
            logger.warning(f"Unknown symptoms (will be ignored): {unknown_symptoms}")

        return vector

    def predict(
        self,
        symptom_vector: np.ndarray,
        top_k: int = 10
    ) -> PredictionResult:
        """
        Predict disease from symptom vector.

        Args:
            symptom_vector: Binary array of shape (num_symptoms,)
            top_k: Number of top predictions to return

        Returns:
            PredictionResult object
        """
        # Convert to tensor
        x = torch.FloatTensor(symptom_vector).unsqueeze(0).to(self.device)

        # Get prediction
        with torch.no_grad():
            output = self.model(x)
            probs = torch.softmax(output, dim=1)

            # Get top-k predictions
            top_probs, top_indices = torch.topk(probs, k=min(top_k, self.num_classes), dim=1)

        # Convert to numpy
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        all_probs = probs.cpu().numpy()[0]

        # Get disease names
        top_diseases = self.label_encoder.inverse_transform(top_indices)

        # Create predictions list
        predictions = list(zip(top_diseases, top_probs))

        # Determine confidence category
        top_confidence = top_probs[0]
        if top_confidence > 0.85:
            confidence_category = "HIGH"
            needs_questioning = False
        elif top_confidence > 0.60:
            confidence_category = "MEDIUM"
            needs_questioning = True
        else:
            confidence_category = "LOW"
            needs_questioning = True

        # Create hypotheses for active questioning
        hypotheses = [
            Hypothesis(
                disease_name=disease,
                disease_id=int(self.label_encoder.transform([disease])[0]),
                probability=float(prob),
                confidence_score=float(prob)
            )
            for disease, prob in predictions
        ]

        return PredictionResult(
            top_disease=top_diseases[0],
            top_confidence=float(top_confidence),
            top_k_predictions=[(d, float(p)) for d, p in predictions],
            all_probabilities=all_probs,
            confidence_category=confidence_category,
            needs_questioning=needs_questioning,
            hypotheses=hypotheses
        )

    def predict_from_symptoms(
        self,
        symptom_names: List[str],
        top_k: int = 10
    ) -> PredictionResult:
        """
        Predict disease from symptom names.

        Args:
            symptom_names: List of symptom names (Vietnamese)
            top_k: Number of top predictions to return

        Returns:
            PredictionResult object
        """
        symptom_vector = self.create_symptom_vector(symptom_names)
        return self.predict(symptom_vector, top_k=top_k)

    def start_questioning_session(
        self,
        session_id: str,
        initial_symptoms: List[str]
    ) -> Tuple[QuestioningSession, Optional[Question]]:
        """
        Start an active questioning session.

        Args:
            session_id: Unique session identifier
            initial_symptoms: List of initial symptoms

        Returns:
            Tuple of (session, first_question)
        """
        # Get initial prediction
        prediction = self.predict_from_symptoms(initial_symptoms, top_k=10)

        # Create questioning session
        session = self.questioning_engine.create_session(
            session_id=session_id,
            initial_symptoms=initial_symptoms,
            initial_hypotheses=prediction.hypotheses
        )

        # Store session
        self.active_sessions[session_id] = session

        # Check if we should question
        should_continue, reason = self.questioning_engine.should_continue_questioning(
            session.hypotheses, session.questions_asked
        )

        if not should_continue:
            session.is_complete = True
            session.final_diagnosis = session.hypotheses[0]
            logger.info(f"Session {session_id} complete: {reason}")
            return session, None

        # Get current symptom vector
        current_symptom_vector = self.create_symptom_vector(initial_symptoms, warn_unknown=False)

        # Select first question
        question = self.questioning_engine.select_next_question(
            hypotheses=session.hypotheses,
            disease_symptom_matrix=self.disease_symptom_matrix,
            already_asked=session.asked_symptoms,
            current_symptom_vector=current_symptom_vector
        )

        return session, question

    def answer_question(
        self,
        session_id: str,
        symptom_name: str,
        has_symptom: bool
    ) -> Tuple[QuestioningSession, Optional[Question]]:
        """
        Process answer to a question and get next question.

        Args:
            session_id: Session identifier
            symptom_name: Name of symptom asked about
            has_symptom: True if user has symptom, False otherwise

        Returns:
            Tuple of (updated_session, next_question)
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]

        # Record answer
        session.symptom_responses[symptom_name] = has_symptom
        session.asked_symptoms.add(symptom_name)
        session.questions_asked += 1

        # Update symptom vector
        all_symptoms = session.initial_symptoms.copy()
        for symptom, present in session.symptom_responses.items():
            if present and symptom not in all_symptoms:
                all_symptoms.append(symptom)

        # Re-predict with updated symptoms
        prediction = self.predict_from_symptoms(all_symptoms, top_k=10)
        session.hypotheses = prediction.hypotheses

        logger.info(f"Session {session_id}: Q{session.questions_asked} - {symptom_name} = {has_symptom}")
        logger.info(f"Top disease now: {prediction.top_disease} ({prediction.top_confidence*100:.1f}%)")

        # Check if we should continue
        should_continue, reason = self.questioning_engine.should_continue_questioning(
            session.hypotheses, session.questions_asked
        )

        if not should_continue:
            session.is_complete = True
            session.final_diagnosis = session.hypotheses[0]
            logger.info(f"Session {session_id} complete: {reason}")
            return session, None

        # Get current symptom vector
        current_symptom_vector = self.create_symptom_vector(all_symptoms, warn_unknown=False)

        # Select next question
        next_question = self.questioning_engine.select_next_question(
            hypotheses=session.hypotheses,
            disease_symptom_matrix=self.disease_symptom_matrix,
            already_asked=session.asked_symptoms,
            current_symptom_vector=current_symptom_vector
        )

        return session, next_question

    def get_session(self, session_id: str) -> Optional[QuestioningSession]:
        """Get disease-based questioning session by ID"""
        return self.active_sessions.get(session_id)

    def end_session(self, session_id: str) -> Optional[QuestioningSession]:
        """End and remove disease-based questioning session"""
        return self.active_sessions.pop(session_id, None)

    # ========== DEPARTMENT-BASED QUESTIONING METHODS ==========

    def start_department_questioning_session(
        self,
        session_id: str,
        initial_symptoms: List[str]
    ) -> Tuple[DepartmentQuestioningSession, Optional[DepartmentQuestion]]:
        """
        Start a department-based questioning session.

        Args:
            session_id: Unique session identifier
            initial_symptoms: List of initial symptoms

        Returns:
            Tuple of (session, first_question)
        """
        if not self.dept_questioning_engine:
            raise ValueError("Department questioning engine not initialized (no department map provided)")

        # Get initial prediction
        prediction = self.predict_from_symptoms(initial_symptoms, top_k=10)

        # Get initial symptom vector
        initial_symptom_vector = self.create_symptom_vector(initial_symptoms, warn_unknown=False)

        # Start department questioning session
        session, first_question = self.dept_questioning_engine.start_questioning_session(
            session_id=session_id,
            disease_probs=prediction.all_probabilities,
            initial_symptom_vector=initial_symptom_vector
        )

        # Store session
        self.dept_active_sessions[session_id] = session

        logger.info(f"Started department questioning session: {session_id}")
        if session.current_departments:
            logger.info(f"Top department: {session.current_departments[0].department_name} "
                       f"({session.current_departments[0].probability*100:.1f}%)")

        return session, first_question

    def answer_department_question(
        self,
        session_id: str,
        symptom_name: str,
        has_symptom: bool
    ) -> Tuple[DepartmentQuestioningSession, Optional[DepartmentQuestion]]:
        """
        Process answer to a department-based question and get next question.

        Args:
            session_id: Session identifier
            symptom_name: Name of symptom asked about
            has_symptom: True if user has symptom, False otherwise

        Returns:
            Tuple of (updated_session, next_question)
        """
        if session_id not in self.dept_active_sessions:
            raise ValueError(f"Department session {session_id} not found")

        if not self.dept_questioning_engine:
            raise ValueError("Department questioning engine not initialized")

        session = self.dept_active_sessions[session_id]

        # Build current symptom list
        all_symptoms = []
        for symptom_idx, symptom_val in enumerate(session.current_symptom_vector):
            if symptom_val == 1:
                all_symptoms.append(self.symptom_columns[symptom_idx])

        # Add the answered symptom
        if has_symptom and symptom_name not in all_symptoms:
            all_symptoms.append(symptom_name)

        # Re-predict with updated symptoms
        prediction = self.predict_from_symptoms(all_symptoms, top_k=10)

        # Update department questioning session
        updated_session, next_question = self.dept_questioning_engine.answer_question(
            session_id=session_id,
            symptom_name=symptom_name,
            has_symptom=has_symptom,
            disease_probs_updated=prediction.all_probabilities
        )

        logger.info(f"Department session {session_id}: Q{updated_session.questions_asked} "
                   f"- {symptom_name} = {has_symptom}")
        if updated_session.current_departments:
            logger.info(f"Top department now: {updated_session.current_departments[0].department_name} "
                       f"({updated_session.current_departments[0].probability*100:.1f}%)")

        return updated_session, next_question

    def get_department_session(self, session_id: str) -> Optional[DepartmentQuestioningSession]:
        """Get department-based questioning session by ID"""
        return self.dept_active_sessions.get(session_id)

    def end_department_session(self, session_id: str) -> Optional[DepartmentQuestioningSession]:
        """End and remove department-based questioning session"""
        return self.dept_active_sessions.pop(session_id, None)

    def get_disease_info(self, disease_name: str) -> Dict:
        """
        Get information about a specific disease.

        Args:
            disease_name: Name of disease

        Returns:
            Dictionary with disease information
        """
        try:
            disease_id = int(self.label_encoder.transform([disease_name])[0])
            disease_symptoms_vector = self.disease_symptom_matrix[disease_id]

            # Get common symptoms (> 50% frequency)
            common_symptom_indices = np.where(disease_symptoms_vector > 0.5)[0]
            common_symptoms = [self.idx_to_symptom[idx] for idx in common_symptom_indices]

            # Get possible symptoms (> 10% frequency)
            possible_symptom_indices = np.where(disease_symptoms_vector > 0.1)[0]
            possible_symptoms = [self.idx_to_symptom[idx] for idx in possible_symptom_indices]

            return {
                "disease_name": disease_name,
                "disease_id": disease_id,
                "common_symptoms": common_symptoms,
                "possible_symptoms": possible_symptoms,
                "num_training_samples": int((self.train_df['disease'] == disease_name).sum())
            }
        except Exception as e:
            logger.error(f"Error getting disease info for {disease_name}: {e}")
            return {
                "disease_name": disease_name,
                "error": str(e)
            }
