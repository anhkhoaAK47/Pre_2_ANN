"""
Active Questioning Module for Medical Diagnosis System

This module implements advanced active learning strategies to iteratively
gather symptom information and refine disease predictions through:
- Information gain calculation
- Entropy-based question selection
- Differential diagnosis for confused pairs
- Bayesian probability updating

Author: AI Medical Chatbot Team
Date: 2025-10-18
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QuestioningStrategy(Enum):
    """Strategy for selecting next question"""
    DIFFERENTIAL = "differential"  # Ask differentiating questions for confused pairs
    INFORMATION_GAIN = "information_gain"  # Maximize information gain
    ENTROPY = "entropy"  # Minimize expected entropy
    HYBRID = "hybrid"  # Combine all strategies


@dataclass
class Hypothesis:
    """Represents a disease hypothesis with probability"""
    disease_name: str
    disease_id: int
    probability: float
    confidence_score: float


@dataclass
class Question:
    """Represents a question to ask the user"""
    symptom_name: str
    symptom_id: int
    question_text: str
    reason: str
    information_gain: float
    strategy_used: QuestioningStrategy


@dataclass
class QuestioningSession:
    """Manages an active questioning session"""
    session_id: str
    initial_symptoms: List[str]
    asked_symptoms: Set[str]
    symptom_responses: Dict[str, bool]  # symptom_name -> yes/no
    hypotheses: List[Hypothesis]
    questions_asked: int
    max_questions: int
    confidence_threshold: float
    is_complete: bool
    final_diagnosis: Optional[Hypothesis]


class ActiveQuestioningEngine:
    """
    Advanced active questioning engine using information theory
    and differential diagnosis strategies.
    """

    def __init__(
        self,
        symptom_columns: List[str],
        confusion_matrix_data: Optional[pd.DataFrame] = None,
        max_questions: int = 10,
        confidence_threshold: float = 0.85,
        gap_threshold: float = 0.40,
        strategy: QuestioningStrategy = QuestioningStrategy.HYBRID
    ):
        """
        Initialize the active questioning engine.

        Args:
            symptom_columns: List of all 377 symptom names
            confusion_matrix_data: DataFrame with confused disease pairs
            max_questions: Maximum questions to ask (default: 10)
            confidence_threshold: Stop when top disease > this (default: 0.85)
            gap_threshold: Stop when gap between #1 and #2 > this (default: 0.40)
            strategy: Questioning strategy to use
        """
        self.symptom_columns = symptom_columns
        self.num_symptoms = len(symptom_columns)
        self.confusion_matrix_data = confusion_matrix_data
        self.max_questions = max_questions
        self.confidence_threshold = confidence_threshold
        self.gap_threshold = gap_threshold
        self.strategy = strategy

        # Create symptom name to index mapping
        self.symptom_to_idx = {symptom: idx for idx, symptom in enumerate(symptom_columns)}
        self.idx_to_symptom = {idx: symptom for symptom, idx in self.symptom_to_idx.items()}

        # Load question cache
        self.question_cache = self._load_question_cache()

        # Load Gemini configuration from environment
        import os
        from dotenv import load_dotenv
        load_dotenv()

        self.enable_gemini = os.getenv("ENABLE_GEMINI_QUESTIONS", "False").lower() == "true"
        self.max_gemini_questions = int(os.getenv("MAX_GEMINI_QUESTIONS", "3"))
        self.gemini_timeout = float(os.getenv("GEMINI_QUESTION_TIMEOUT", "2.0"))

        logger.info(f"ActiveQuestioningEngine initialized with {self.num_symptoms} symptoms")
        logger.info(f"Strategy: {strategy.value}, Max questions: {max_questions}")
        if self.question_cache:
            logger.info(f"Loaded question cache with {len(self.question_cache)} symptoms")
        logger.info(f"Gemini questions: {'Enabled' if self.enable_gemini else 'Disabled'} "
                   f"(max: {self.max_gemini_questions}, timeout: {self.gemini_timeout}s)")

    def should_continue_questioning(
        self,
        hypotheses: List[Hypothesis],
        questions_asked: int
    ) -> Tuple[bool, str]:
        """
        Determine if we should continue asking questions.

        Args:
            hypotheses: Current disease hypotheses with probabilities
            questions_asked: Number of questions asked so far

        Returns:
            Tuple of (should_continue, reason)
        """
        # Check max questions
        if questions_asked >= self.max_questions:
            return False, f"Maximum questions reached ({self.max_questions})"

        # Check if we have hypotheses
        if not hypotheses or len(hypotheses) == 0:
            return False, "No valid hypotheses"

        # Sort by probability
        sorted_hyp = sorted(hypotheses, key=lambda h: h.probability, reverse=True)
        top_prob = sorted_hyp[0].probability

        # Check confidence threshold
        if top_prob >= self.confidence_threshold:
            return False, f"High confidence reached ({top_prob*100:.1f}% > {self.confidence_threshold*100:.0f}%)"

        # Check gap between top 2
        if len(sorted_hyp) >= 2:
            gap = sorted_hyp[0].probability - sorted_hyp[1].probability
            if gap >= self.gap_threshold:
                return False, f"Clear winner (gap: {gap*100:.1f}% > {self.gap_threshold*100:.0f}%)"

        # Continue questioning
        return True, "Confidence still low, continue questioning"

    def calculate_entropy(self, probabilities: np.ndarray) -> float:
        """
        Calculate Shannon entropy of probability distribution.

        Args:
            probabilities: Array of probabilities

        Returns:
            Entropy value
        """
        # Filter out zero probabilities
        probs = probabilities[probabilities > 0]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))

    def calculate_information_gain(
        self,
        symptom_idx: int,
        current_probs: np.ndarray,
        disease_symptom_matrix: np.ndarray
    ) -> float:
        """
        Calculate information gain for asking about a specific symptom.

        Information Gain = H(current) - E[H(after question)]

        Args:
            symptom_idx: Index of symptom to ask about
            current_probs: Current probability distribution over diseases
            disease_symptom_matrix: Matrix of shape (num_diseases, num_symptoms)
                                   where [i,j] = 1 if disease i has symptom j

        Returns:
            Information gain value
        """
        # Current entropy
        current_entropy = self.calculate_entropy(current_probs)

        # Calculate P(symptom = YES) and P(symptom = NO)
        p_yes = np.sum(current_probs * disease_symptom_matrix[:, symptom_idx])
        p_no = 1 - p_yes

        # If probability is 0 or 1, no information gain
        if p_yes == 0 or p_no == 0:
            return 0.0

        # Calculate updated probabilities if answer is YES
        probs_if_yes = current_probs * disease_symptom_matrix[:, symptom_idx]
        if probs_if_yes.sum() > 0:
            probs_if_yes = probs_if_yes / probs_if_yes.sum()
        entropy_if_yes = self.calculate_entropy(probs_if_yes)

        # Calculate updated probabilities if answer is NO
        probs_if_no = current_probs * (1 - disease_symptom_matrix[:, symptom_idx])
        if probs_if_no.sum() > 0:
            probs_if_no = probs_if_no / probs_if_no.sum()
        entropy_if_no = self.calculate_entropy(probs_if_no)

        # Expected entropy after asking question
        expected_entropy = p_yes * entropy_if_yes + p_no * entropy_if_no

        # Information gain
        info_gain = current_entropy - expected_entropy

        return info_gain

    def find_differential_symptoms(
        self,
        disease1_id: int,
        disease2_id: int,
        disease_symptom_matrix: np.ndarray,
        already_asked: Set[str]
    ) -> List[Tuple[int, str, float]]:
        """
        Find symptoms that differentiate between two diseases.

        Args:
            disease1_id: ID of first disease
            disease2_id: ID of second disease
            disease_symptom_matrix: Matrix of disease-symptom associations
            already_asked: Set of symptoms already asked

        Returns:
            List of (symptom_idx, symptom_name, discriminative_power)
        """
        differential_symptoms = []

        d1_symptoms = disease_symptom_matrix[disease1_id]
        d2_symptoms = disease_symptom_matrix[disease2_id]

        for idx in range(self.num_symptoms):
            symptom_name = self.idx_to_symptom[idx]

            # Skip if already asked
            if symptom_name in already_asked:
                continue

            # Check if symptom differs between diseases
            if d1_symptoms[idx] != d2_symptoms[idx]:
                # Discriminative power = absolute difference
                discriminative_power = abs(d1_symptoms[idx] - d2_symptoms[idx])
                differential_symptoms.append((idx, symptom_name, discriminative_power))

        # Sort by discriminative power
        differential_symptoms.sort(key=lambda x: x[2], reverse=True)

        return differential_symptoms

    def select_next_question(
        self,
        hypotheses: List[Hypothesis],
        disease_symptom_matrix: np.ndarray,
        already_asked: Set[str],
        current_symptom_vector: np.ndarray
    ) -> Optional[Question]:
        """
        Select the next best question to ask using the chosen strategy.

        Args:
            hypotheses: Current disease hypotheses
            disease_symptom_matrix: Matrix of disease-symptom associations
            already_asked: Set of symptoms already asked
            current_symptom_vector: Current symptom vector (binary)

        Returns:
            Question object or None if no good question found
        """
        if not hypotheses:
            return None

        # Get current probability distribution
        sorted_hyp = sorted(hypotheses, key=lambda h: h.probability, reverse=True)

        # Extract top disease names for context
        top_disease_names = [h.disease_name for h in sorted_hyp[:3]]

        # Create probability vector for all diseases
        full_probs = np.zeros(disease_symptom_matrix.shape[0])
        for hyp in hypotheses:
            full_probs[hyp.disease_id] = hyp.probability

        best_question = None
        best_score = -1

        # Strategy 1: Differential Diagnosis (for top 2 diseases)
        if self.strategy in [QuestioningStrategy.DIFFERENTIAL, QuestioningStrategy.HYBRID]:
            if len(sorted_hyp) >= 2:
                top1_id = sorted_hyp[0].disease_id
                top2_id = sorted_hyp[1].disease_id

                diff_symptoms = self.find_differential_symptoms(
                    top1_id, top2_id, disease_symptom_matrix, already_asked
                )

                if diff_symptoms and len(diff_symptoms) > 0:
                    symptom_idx, symptom_name, disc_power = diff_symptoms[0]

                    # Only use if discriminative power is high
                    if disc_power > 0.5:
                        best_question = Question(
                            symptom_name=symptom_name,
                            symptom_id=symptom_idx,
                            question_text=self._create_question_text(
                                symptom_name,
                                top_diseases=top_disease_names
                            ),
                            reason="Để phân biệt giữa các bệnh có khả năng cao",
                            information_gain=disc_power,
                            strategy_used=QuestioningStrategy.DIFFERENTIAL
                        )
                        best_score = disc_power

        # Strategy 2: Information Gain (general approach)
        if self.strategy in [QuestioningStrategy.INFORMATION_GAIN, QuestioningStrategy.HYBRID]:
            candidate_symptoms = []

            for idx in range(self.num_symptoms):
                symptom_name = self.idx_to_symptom[idx]

                # Skip if already asked or already present
                if symptom_name in already_asked or current_symptom_vector[idx] == 1:
                    continue

                # Calculate information gain
                info_gain = self.calculate_information_gain(
                    idx, full_probs, disease_symptom_matrix
                )

                candidate_symptoms.append((idx, symptom_name, info_gain))

            # Sort by information gain
            candidate_symptoms.sort(key=lambda x: x[2], reverse=True)

            # Select best if better than current best
            if candidate_symptoms and candidate_symptoms[0][2] > best_score:
                symptom_idx, symptom_name, info_gain = candidate_symptoms[0]
                best_question = Question(
                    symptom_name=symptom_name,
                    symptom_id=symptom_idx,
                    question_text=self._create_question_text(
                        symptom_name,
                        top_diseases=top_disease_names
                    ),
                    reason="Để thu thập thêm thông tin về các triệu chứng",
                    information_gain=info_gain,
                    strategy_used=QuestioningStrategy.INFORMATION_GAIN
                )
                best_score = info_gain

        # Strategy 3: Entropy Reduction (similar to information gain but explicit)
        if self.strategy == QuestioningStrategy.ENTROPY and best_score < 0:
            # This is already covered by information gain, so we can skip
            # But keeping the structure for future enhancements
            pass

        return best_question

    def _create_question_text(
        self,
        symptom_name: str,
        question_number: int = 1,
        top_diseases: List[str] = None
    ) -> str:
        """
        Create natural language question from symptom name.
        Strategy: Gemini (if enabled) → Cache → Templates

        Args:
            symptom_name: Vietnamese symptom name
            question_number: Question number in session (for Gemini context)
            top_diseases: Top disease names (for Gemini context)

        Returns:
            Natural language question
        """
        import random

        # Strategy 1: Try Gemini (if enabled and within limit)
        if self.enable_gemini and question_number <= self.max_gemini_questions:
            try:
                from app_v2.services.gemini_service import generate_diverse_question
                question = generate_diverse_question(
                    symptom_name=symptom_name,
                    question_number=question_number,
                    top_diseases=top_diseases,
                    timeout=self.gemini_timeout
                )
                logger.debug(f"Generated Gemini question for '{symptom_name}': {question}")
                return question
            except Exception as e:
                logger.warning(f"Gemini question generation failed for '{symptom_name}': {e}, "
                             f"falling back to cache/template")

        # Strategy 2: Try cache
        if self.question_cache and symptom_name in self.question_cache:
            return random.choice(self.question_cache[symptom_name])

        # Strategy 3: Fallback templates
        templates = [
            f"Bạn có bị {symptom_name} không?",
            f"Bạn có cảm thấy {symptom_name} không?",
            f"Gần đây bạn có bị {symptom_name} không?",
            f"Hiện tại bạn có {symptom_name} không?",
            f"Bạn đang bị {symptom_name} phải không?",
            f"Cho tôi hỏi, bạn có {symptom_name} không?",
            f"Bạn có triệu chứng {symptom_name} không?"
        ]

        return random.choice(templates)

    def _load_question_cache(self) -> Dict[str, List[str]]:
        """
        Load question cache from JSON file.

        Returns:
            Dictionary mapping symptom names to lists of question variations
        """
        import json
        import os

        try:
            # Get path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            cache_path = os.path.join(current_dir, "..", "data", "question_cache.json")
            cache_path = os.path.normpath(cache_path)

            if not os.path.exists(cache_path):
                logger.warning(f"Question cache not found at {cache_path}")
                return {}

            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract questions from both system and user sections
            cache = {}

            # Load system generated questions
            if "SYSTEM_GENERATED_QUESTIONS" in data:
                cache.update(data["SYSTEM_GENERATED_QUESTIONS"])

            # Load user custom questions (skip entries starting with _)
            if "USER_CUSTOM_QUESTIONS" in data:
                user_questions = data["USER_CUSTOM_QUESTIONS"]
                for key, value in user_questions.items():
                    # Skip instruction entries (start with _)
                    if not key.startswith("_") and isinstance(value, list):
                        cache[key] = value

            logger.info(f"Successfully loaded question cache from {cache_path}")
            return cache

        except Exception as e:
            logger.error(f"Failed to load question cache: {e}")
            return {}

    def update_probabilities(
        self,
        current_probs: List[float],
        symptom_idx: int,
        symptom_present: bool,
        disease_symptom_matrix: np.ndarray,
        disease_ids: List[int]
    ) -> List[float]:
        """
        Update disease probabilities using Bayesian update after answer.

        Args:
            current_probs: Current probabilities for candidate diseases
            symptom_idx: Index of symptom asked about
            symptom_present: True if user has symptom, False otherwise
            disease_symptom_matrix: Matrix of disease-symptom associations
            disease_ids: List of disease IDs corresponding to current_probs

        Returns:
            Updated probabilities
        """
        current_probs = np.array(current_probs)

        if symptom_present:
            # Update: P(disease | symptom=yes) ∝ P(disease) * P(symptom=yes | disease)
            likelihood = np.array([disease_symptom_matrix[did, symptom_idx] for did in disease_ids])
        else:
            # Update: P(disease | symptom=no) ∝ P(disease) * P(symptom=no | disease)
            likelihood = np.array([1 - disease_symptom_matrix[did, symptom_idx] for did in disease_ids])

        # Bayesian update
        updated_probs = current_probs * likelihood

        # Normalize
        total = updated_probs.sum()
        if total > 0:
            updated_probs = updated_probs / total
        else:
            # If all probabilities become 0, revert to uniform
            updated_probs = np.ones(len(current_probs)) / len(current_probs)

        return updated_probs.tolist()

    def create_session(
        self,
        session_id: str,
        initial_symptoms: List[str],
        initial_hypotheses: List[Hypothesis]
    ) -> QuestioningSession:
        """
        Create a new active questioning session.

        Args:
            session_id: Unique session identifier
            initial_symptoms: List of initial symptoms provided
            initial_hypotheses: Initial disease hypotheses from model

        Returns:
            QuestioningSession object
        """
        session = QuestioningSession(
            session_id=session_id,
            initial_symptoms=initial_symptoms,
            asked_symptoms=set(initial_symptoms),
            symptom_responses={},
            hypotheses=initial_hypotheses,
            questions_asked=0,
            max_questions=self.max_questions,
            confidence_threshold=self.confidence_threshold,
            is_complete=False,
            final_diagnosis=None
        )

        logger.info(f"Created questioning session {session_id} with {len(initial_symptoms)} initial symptoms")
        logger.info(f"Top hypothesis: {initial_hypotheses[0].disease_name} ({initial_hypotheses[0].probability*100:.1f}%)")

        return session

    def get_summary(self, session: QuestioningSession) -> Dict:
        """
        Get summary of questioning session.

        Args:
            session: QuestioningSession object

        Returns:
            Dictionary with session summary
        """
        return {
            "session_id": session.session_id,
            "questions_asked": session.questions_asked,
            "max_questions": session.max_questions,
            "is_complete": session.is_complete,
            "initial_symptoms_count": len(session.initial_symptoms),
            "additional_symptoms_count": len(session.symptom_responses),
            "total_symptoms": len(session.initial_symptoms) + len(session.symptom_responses),
            "top_hypotheses": [
                {
                    "disease": h.disease_name,
                    "probability": round(h.probability, 4),
                    "confidence": round(h.confidence_score, 4)
                }
                for h in sorted(session.hypotheses, key=lambda h: h.probability, reverse=True)[:5]
            ],
            "final_diagnosis": {
                "disease": session.final_diagnosis.disease_name,
                "probability": round(session.final_diagnosis.probability, 4),
                "confidence": round(session.final_diagnosis.confidence_score, 4)
            } if session.final_diagnosis else None
        }


def create_disease_symptom_matrix(
    train_df: pd.DataFrame,
    label_encoder
) -> Tuple[np.ndarray, List[str]]:
    """
    Create disease-symptom association matrix from training data.

    This matrix represents which symptoms are associated with which diseases
    based on the training data.

    Args:
        train_df: Training DataFrame with disease and symptom columns
        label_encoder: LabelEncoder for disease names

    Returns:
        Tuple of (matrix, symptom_columns)
        matrix: Array of shape (num_diseases, num_symptoms)
        symptom_columns: List of symptom column names
    """
    symptom_columns = [col for col in train_df.columns if col != 'disease']
    num_diseases = len(label_encoder.classes_)
    num_symptoms = len(symptom_columns)

    # Initialize matrix
    matrix = np.zeros((num_diseases, num_symptoms), dtype=np.float32)

    # Fill matrix with symptom frequencies for each disease
    for disease_idx, disease_name in enumerate(label_encoder.classes_):
        disease_samples = train_df[train_df['disease'] == disease_name]

        if len(disease_samples) > 0:
            # Calculate frequency of each symptom for this disease
            symptom_frequencies = disease_samples[symptom_columns].mean(axis=0).values
            matrix[disease_idx] = symptom_frequencies

    logger.info(f"Created disease-symptom matrix: {matrix.shape}")
    logger.info(f"Matrix sparsity: {(matrix == 0).sum() / matrix.size * 100:.2f}%")

    return matrix, symptom_columns
