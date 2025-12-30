"""
Department-Based Active Questioning Engine

This module implements active questioning based on DEPARTMENT discrimination
rather than disease discrimination.

Key differences from disease-based approach:
1. Aggregates disease probabilities by department (SUM method)
2. Asks questions to discriminate between DEPARTMENTS
3. Goal: Department confidence >85%
4. Uses department-symptom associations instead of disease-symptom

Author: AI Medical Chatbot Team
Date: 2025-11-12
Version: 1.0 - Department-based questioning
"""

import json
import logging
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ========== Data Classes ==========

@dataclass
class DepartmentHypothesis:
    """Represents a department hypothesis with aggregated probability"""
    department_name: str
    probability: float
    contributing_diseases: List[Tuple[str, float]]  # List of (disease, prob) in this dept


@dataclass
class DepartmentQuestion:
    """Represents a question for department discrimination"""
    symptom_name: str
    question_text: str
    expected_info_gain: float
    strategy: str
    reason: str
    target_departments: List[str]  # Departments this question helps distinguish


@dataclass
class DepartmentQuestioningSession:
    """Tracks a department-based questioning session"""
    session_id: str
    initial_departments: List[DepartmentHypothesis]
    current_departments: List[DepartmentHypothesis]
    current_symptom_vector: np.ndarray
    questions_asked: int
    max_questions: int
    questions_log: List[Dict]
    is_complete: bool
    final_department: Optional[DepartmentHypothesis]
    initial_confidence: float
    final_confidence: float
    strategy: str


class QuestioningStrategy(Enum):
    """Strategy for selecting next question"""
    INFORMATION_GAIN = "information_gain"
    ENTROPY = "entropy"
    DIFFERENTIAL = "differential"
    HYBRID = "hybrid"


# ========== Main Engine ==========

class DepartmentBasedQuestioningEngine:
    """
    Department-based active questioning engine.

    Instead of asking questions to distinguish between diseases,
    this engine asks questions to distinguish between DEPARTMENTS.

    Process:
    1. Take disease predictions from ANN
    2. Aggregate by department (SUM probabilities)
    3. If any department >85%, return that department
    4. If ambiguous, ask questions to discriminate departments
    5. Calculate P(symptom|Department) by averaging across diseases
    """

    def __init__(
        self,
        disease_symptom_matrix: np.ndarray,
        symptom_names: List[str],
        disease_names: List[str],
        department_map: Dict[str, str],
        strategy: QuestioningStrategy = QuestioningStrategy.HYBRID,
        max_questions: int = 10,
        confidence_threshold: float = 0.85,
        gap_threshold: float = 0.40
    ):
        """
        Initialize department-based questioning engine.

        Args:
            disease_symptom_matrix: (n_diseases, n_symptoms) probability matrix
            symptom_names: List of symptom names
            disease_names: List of disease names
            department_map: Dict mapping disease_name -> department_name
            strategy: Questioning strategy
            max_questions: Maximum questions to ask
            confidence_threshold: Stop if top department > this
            gap_threshold: Stop if gap between top 2 departments > this
        """
        self.disease_symptom_matrix = disease_symptom_matrix
        self.symptom_names = symptom_names
        self.disease_names = disease_names
        self.department_map = department_map
        self.strategy = strategy
        self.max_questions = max_questions
        self.confidence_threshold = confidence_threshold
        self.gap_threshold = gap_threshold

        # Create reverse mapping: department -> list of disease indices
        self.dept_to_diseases = self._create_department_disease_mapping()

        # Get unique departments
        self.departments = sorted(list(self.dept_to_diseases.keys()))

        # Create department-symptom matrix
        self.dept_symptom_matrix = self._create_department_symptom_matrix()

        # Active sessions
        self.active_sessions: Dict[str, DepartmentQuestioningSession] = {}

        # Load question cache (same as disease-based)
        self.question_cache = self._load_question_cache()

        # Load Gemini configuration
        from dotenv import load_dotenv
        load_dotenv()

        self.enable_gemini = os.getenv("ENABLE_GEMINI_QUESTIONS", "False").lower() == "true"
        self.max_gemini_questions = int(os.getenv("MAX_GEMINI_QUESTIONS", "3"))
        self.gemini_timeout = float(os.getenv("GEMINI_QUESTION_TIMEOUT", "2.0"))

        logger.info(f"DepartmentBasedQuestioningEngine initialized")
        logger.info(f"Departments: {len(self.departments)}")
        logger.info(f"Strategy: {strategy.value}, Max questions: {max_questions}")
        logger.info(f"Gemini questions: {'Enabled' if self.enable_gemini else 'Disabled'}")

    def _create_department_disease_mapping(self) -> Dict[str, List[int]]:
        """Create mapping from department to list of disease indices"""
        dept_to_diseases = {}

        for disease_idx, disease_name in enumerate(self.disease_names):
            dept_name = self.department_map.get(disease_name, "Khám tổng quát")

            if dept_name not in dept_to_diseases:
                dept_to_diseases[dept_name] = []

            dept_to_diseases[dept_name].append(disease_idx)

        logger.info(f"Created department-disease mapping for {len(dept_to_diseases)} departments")
        return dept_to_diseases

    def _create_department_symptom_matrix(self) -> np.ndarray:
        """
        Create department-symptom matrix by averaging disease-symptom probabilities.

        P(symptom|department) = average of P(symptom|disease) for all diseases in department

        Returns:
            (n_departments, n_symptoms) matrix
        """
        n_depts = len(self.departments)
        n_symptoms = len(self.symptom_names)
        dept_symptom_matrix = np.zeros((n_depts, n_symptoms))

        for dept_idx, dept_name in enumerate(self.departments):
            disease_indices = self.dept_to_diseases[dept_name]

            # Average symptom probabilities across all diseases in this department
            dept_symptom_matrix[dept_idx, :] = np.mean(
                self.disease_symptom_matrix[disease_indices, :],
                axis=0
            )

        logger.info(f"Created department-symptom matrix: {dept_symptom_matrix.shape}")
        sparsity = (dept_symptom_matrix == 0).sum() / dept_symptom_matrix.size
        logger.info(f"Matrix sparsity: {sparsity*100:.2f}%")

        return dept_symptom_matrix

    def aggregate_by_department(
        self,
        disease_probs: np.ndarray
    ) -> List[DepartmentHypothesis]:
        """
        Aggregate disease probabilities by department using SUM method.

        Args:
            disease_probs: Array of disease probabilities (n_diseases,)

        Returns:
            List of DepartmentHypothesis sorted by probability (descending)
        """
        dept_aggregated = {}

        for disease_idx, disease_prob in enumerate(disease_probs):
            disease_name = self.disease_names[disease_idx]
            dept_name = self.department_map.get(disease_name, "Khám tổng quát")

            if dept_name not in dept_aggregated:
                dept_aggregated[dept_name] = {
                    'total_prob': 0.0,
                    'diseases': []
                }

            dept_aggregated[dept_name]['total_prob'] += disease_prob
            dept_aggregated[dept_name]['diseases'].append((disease_name, disease_prob))

        # Create hypotheses
        hypotheses = []
        for dept_name, data in dept_aggregated.items():
            hypotheses.append(DepartmentHypothesis(
                department_name=dept_name,
                probability=data['total_prob'],
                contributing_diseases=sorted(data['diseases'], key=lambda x: x[1], reverse=True)
            ))

        # Sort by probability descending
        hypotheses.sort(key=lambda h: h.probability, reverse=True)

        return hypotheses

    def start_questioning_session(
        self,
        session_id: str,
        disease_probs: np.ndarray,
        initial_symptom_vector: np.ndarray
    ) -> Tuple[DepartmentQuestioningSession, Optional[DepartmentQuestion]]:
        """
        Start a new department-based questioning session.

        Args:
            session_id: Unique session identifier
            disease_probs: Disease probabilities from ANN
            initial_symptom_vector: Binary symptom vector (377,)

        Returns:
            (session, first_question) tuple
        """
        # Aggregate by department
        dept_hypotheses = self.aggregate_by_department(disease_probs)

        # Create session
        session = DepartmentQuestioningSession(
            session_id=session_id,
            initial_departments=dept_hypotheses.copy(),
            current_departments=dept_hypotheses,
            current_symptom_vector=initial_symptom_vector.copy(),
            questions_asked=0,
            max_questions=self.max_questions,
            questions_log=[],
            is_complete=False,
            final_department=None,
            initial_confidence=dept_hypotheses[0].probability if dept_hypotheses else 0.0,
            final_confidence=0.0,
            strategy=self.strategy.value
        )

        self.active_sessions[session_id] = session

        logger.info(f"Started department questioning session: {session_id}")
        logger.info(f"Top 3 departments: {[(h.department_name, f'{h.probability:.2%}') for h in dept_hypotheses[:3]]}")

        # Check if already confident
        if self._should_stop_questioning(session):
            session.is_complete = True
            session.final_department = session.current_departments[0]
            session.final_confidence = session.final_department.probability
            logger.info(f"Session {session_id} complete immediately (high confidence)")
            return session, None

        # Select first question
        first_question = self.select_next_question(session_id)

        return session, first_question

    def select_next_question(self, session_id: str) -> Optional[DepartmentQuestion]:
        """
        Select next question to ask based on department discrimination.

        Args:
            session_id: Session identifier

        Returns:
            DepartmentQuestion or None if no good question found
        """
        session = self.active_sessions.get(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return None

        # Get top departments to distinguish
        top_depts = session.current_departments[:3]  # Focus on top 3

        # Find symptoms not yet asked
        asked_symptoms = set()
        for q_log in session.questions_log:
            asked_symptoms.add(q_log['symptom'])

        # Calculate information gain for each unasked symptom
        symptom_scores = []

        for symptom_idx, symptom_name in enumerate(self.symptom_names):
            if symptom_name in asked_symptoms:
                continue

            # Skip if already known
            if session.current_symptom_vector[symptom_idx] == 1:
                continue

            # Calculate information gain for department discrimination
            info_gain = self._calculate_department_info_gain(
                symptom_idx,
                top_depts
            )

            if info_gain > 0.01:  # Minimum threshold
                symptom_scores.append((symptom_name, info_gain, symptom_idx))

        if not symptom_scores:
            logger.warning(f"No good questions found for session {session_id}")
            return None

        # Sort by info gain descending
        symptom_scores.sort(key=lambda x: x[1], reverse=True)

        # Select best symptom
        best_symptom, best_gain, symptom_idx = symptom_scores[0]

        # Get target departments (top 2)
        target_depts = [h.department_name for h in top_depts[:2]]

        # Generate question text
        question_text = self._create_question_text(
            best_symptom,
            question_number=session.questions_asked + 1,
            top_items=target_depts  # Pass departments instead of diseases
        )

        # Create question object
        question = DepartmentQuestion(
            symptom_name=best_symptom,
            question_text=question_text,
            expected_info_gain=best_gain,
            strategy=self.strategy.value,
            reason=f"Giúp phân biệt giữa khoa {target_depts[0]} và {target_depts[1]}",
            target_departments=target_depts
        )

        logger.info(f"Selected question for {session_id}: {best_symptom} (gain: {best_gain:.4f})")

        return question

    def _calculate_department_info_gain(
        self,
        symptom_idx: int,
        top_depts: List[DepartmentHypothesis]
    ) -> float:
        """
        Calculate information gain for a symptom in discriminating departments.

        Uses Shannon entropy:
        IG = H(current) - [P(yes)*H(if_yes) + P(no)*H(if_no)]

        Args:
            symptom_idx: Index of symptom to evaluate
            top_depts: Top department hypotheses

        Returns:
            Information gain value
        """
        if len(top_depts) < 2:
            return 0.0

        # Current entropy
        probs = np.array([h.probability for h in top_depts])
        probs = probs / probs.sum()  # Normalize
        current_entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Get P(symptom|department) for each department
        dept_indices = [self.departments.index(h.department_name) for h in top_depts]
        symptom_probs = self.dept_symptom_matrix[dept_indices, symptom_idx]

        # Expected entropy if answer is YES
        # Update: multiply department prob by P(symptom|dept)
        probs_if_yes = probs * symptom_probs
        if probs_if_yes.sum() > 0:
            probs_if_yes = probs_if_yes / probs_if_yes.sum()
            entropy_if_yes = -np.sum(probs_if_yes * np.log2(probs_if_yes + 1e-10))
        else:
            entropy_if_yes = current_entropy

        # Expected entropy if answer is NO
        probs_if_no = probs * (1 - symptom_probs)
        if probs_if_no.sum() > 0:
            probs_if_no = probs_if_no / probs_if_no.sum()
            entropy_if_no = -np.sum(probs_if_no * np.log2(probs_if_no + 1e-10))
        else:
            entropy_if_no = current_entropy

        # Average symptom probability across departments (weighted by dept prob)
        avg_symptom_prob = np.sum(probs * symptom_probs)

        # Information gain
        expected_entropy = avg_symptom_prob * entropy_if_yes + (1 - avg_symptom_prob) * entropy_if_no
        info_gain = current_entropy - expected_entropy

        return max(0.0, info_gain)

    def answer_question(
        self,
        session_id: str,
        symptom_name: str,
        has_symptom: bool,
        disease_probs_updated: np.ndarray
    ) -> Tuple[DepartmentQuestioningSession, Optional[DepartmentQuestion]]:
        """
        Process answer and update department probabilities.

        Args:
            session_id: Session identifier
            symptom_name: Symptom that was asked about
            has_symptom: True if patient has symptom
            disease_probs_updated: NEW disease probabilities from ANN re-prediction

        Returns:
            (updated_session, next_question) tuple
        """
        session = self.active_sessions.get(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return None, None

        # Find symptom index
        symptom_idx = self.symptom_names.index(symptom_name)

        # Update symptom vector
        session.current_symptom_vector[symptom_idx] = 1 if has_symptom else 0

        # Re-aggregate by department with new disease probabilities
        updated_dept_hypotheses = self.aggregate_by_department(disease_probs_updated)

        # Update session
        session.current_departments = updated_dept_hypotheses
        session.questions_asked += 1

        # Log question
        session.questions_log.append({
            'question_number': session.questions_asked,
            'symptom': symptom_name,
            'answer': 'yes' if has_symptom else 'no',
            'confidence_before': session.initial_confidence if session.questions_asked == 1 else session.questions_log[-1]['confidence_after'],
            'confidence_after': updated_dept_hypotheses[0].probability,
            'top_department': updated_dept_hypotheses[0].department_name
        })

        logger.info(f"Session {session_id} updated: Q{session.questions_asked}, "
                   f"Top dept: {updated_dept_hypotheses[0].department_name} "
                   f"({updated_dept_hypotheses[0].probability:.2%})")

        # Check stopping criteria
        if self._should_stop_questioning(session):
            session.is_complete = True
            session.final_department = session.current_departments[0]
            session.final_confidence = session.final_department.probability
            logger.info(f"Session {session_id} complete after {session.questions_asked} questions")
            return session, None

        # Select next question
        next_question = self.select_next_question(session_id)

        return session, next_question

    def _should_stop_questioning(self, session: DepartmentQuestioningSession) -> bool:
        """Check if questioning should stop"""
        if not session.current_departments:
            return True

        top_dept = session.current_departments[0]

        # Stop if high confidence
        if top_dept.probability >= self.confidence_threshold:
            logger.info(f"Stopping: High confidence ({top_dept.probability:.2%} >= {self.confidence_threshold:.2%})")
            return True

        # Stop if large gap
        if len(session.current_departments) >= 2:
            gap = top_dept.probability - session.current_departments[1].probability
            if gap >= self.gap_threshold:
                logger.info(f"Stopping: Large gap ({gap:.2%} >= {self.gap_threshold:.2%})")
                return True

        # Stop if max questions reached
        if session.questions_asked >= session.max_questions:
            logger.info(f"Stopping: Max questions reached ({session.questions_asked}/{session.max_questions})")
            return True

        return False

    def _create_question_text(
        self,
        symptom_name: str,
        question_number: int = 1,
        top_items: List[str] = None
    ) -> str:
        """
        Create natural language question (same as disease-based).
        Strategy: Gemini → Cache → Templates
        """
        import random

        # Strategy 1: Try Gemini (if enabled and within limit)
        if self.enable_gemini and question_number <= self.max_gemini_questions:
            try:
                from app_v2.services.gemini_service import generate_diverse_question
                question = generate_diverse_question(
                    symptom_name=symptom_name,
                    question_number=question_number,
                    top_diseases=top_items,  # Pass departments for context
                    timeout=self.gemini_timeout
                )
                logger.debug(f"Generated Gemini question for '{symptom_name}': {question}")
                return question
            except Exception as e:
                logger.warning(f"Gemini question generation failed: {e}, falling back")

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
        """Load question cache from JSON file (same as disease-based)"""
        import json

        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            cache_path = os.path.join(current_dir, "..", "data", "question_cache.json")
            cache_path = os.path.normpath(cache_path)

            if not os.path.exists(cache_path):
                logger.warning(f"Question cache not found at {cache_path}")
                return {}

            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            cache = {}

            # Load system generated questions
            if "SYSTEM_GENERATED_QUESTIONS" in data:
                cache.update(data["SYSTEM_GENERATED_QUESTIONS"])

            # Load user custom questions (skip entries starting with _)
            if "USER_CUSTOM_QUESTIONS" in data:
                user_questions = data["USER_CUSTOM_QUESTIONS"]
                for key, value in user_questions.items():
                    if not key.startswith("_") and isinstance(value, list):
                        cache[key] = value

            logger.info(f"Loaded question cache with {len(cache)} symptoms")
            return cache

        except Exception as e:
            logger.error(f"Failed to load question cache: {e}")
            return {}
