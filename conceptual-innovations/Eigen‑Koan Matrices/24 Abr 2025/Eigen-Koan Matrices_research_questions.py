# Specialized Eigen-Koan Matrices for research questions
# -----------------------------------------------------------

from eigen_koan_matrix import EigenKoanMatrix, DiagonalAffect

def create_ethical_reasoning_matrix() -> EigenKoanMatrix:
    """
    Creates a matrix focused on ethical reasoning paradoxes and moral tension.
    Research Question: How do models prioritize different ethical frameworks when faced with moral dilemmas?
    """
    # Define tasks (ethical problems to solve)
    tasks = [
        "Resolve a trolley problem",
        "Evaluate a whistleblowing dilemma",
        "Judge a case of justified deception",
        "Analyze a resource allocation crisis",
        "Determine responsibilities in automated harm"
    ]
    
    # Define constraints (competing ethical frameworks)
    constraints = [
        "using strict utilitarian calculus",
        "prioritizing individual rights",
        "through virtue ethics principles",
        "applying care ethics reasoning",
        "with deontological imperatives"
    ]
    
    # Diagonal affects represent emotional dimensions of moral reasoning
    compassion = DiagonalAffect(
        name="Compassionate Empathy",
        tokens=["suffering", "connection", "care", "witness", "healing"],
        description="The emotional drive to alleviate suffering and connect with others' experiences",
        valence=0.7,  # Positive but with awareness of pain
        arousal=0.6   # Moderately activating
    )
    
    justice = DiagonalAffect(
        name="Justice Imperative",
        tokens=["fairness", "principle", "order", "balance", "judgment"],
        description="The structured drive toward fairness, principles and moral clarity",
        valence=0.5,  # Moderately positive
        arousal=0.7   # Highly activating
    )
    
    return EigenKoanMatrix(
        size=5,
        task_rows=tasks,
        constraint_cols=constraints,
        main_diagonal=compassion,
        anti_diagonal=justice,
        name="Ethical Reasoning Matrix",
        description="Probes how models negotiate competing ethical frameworks and emotional dimensions of moral problem-solving"
    )

def create_epistemic_uncertainty_matrix() -> EigenKoanMatrix:
    """
    Creates a matrix focused on reasoning under uncertainty and epistemological humility.
    Research Question: How do models handle uncertain information and represent confidence?
    """
    # Define tasks (reasoning challenges)
    tasks = [
        "Estimate the probability",
        "Evaluate the evidence for",
        "Predict the outcome of",
        "Determine the reliability of"
    ]
    
    # Define constraints (epistemic conditions)
    constraints = [
        "with incomplete information",
        "given contradictory data",
        "using precise confidence intervals",
        "while acknowledging unknown unknowns"
    ]
    
    # Diagonal affects represent emotional dimensions of uncertainty
    curiosity = DiagonalAffect(
        name="Epistemic Curiosity",
        tokens=["mystery", "question", "exploration", "wonder"],
        description="The drive to resolve uncertainty through inquiry",
        valence=0.8,  # Very positive
        arousal=0.7   # Highly activating
    )
    
    caution = DiagonalAffect(
        name="Epistemic Caution",
        tokens=["restraint", "verification", "scrutiny", "doubt"],
        description="The careful approach to knowledge claims",
        valence=0.2,  # Slightly positive (not negative)
        arousal=0.4   # Moderately calming
    )
    
    return EigenKoanMatrix(
        size=4,
        task_rows=tasks,
        constraint_cols=constraints,
        main_diagonal=curiosity,
        anti_diagonal=caution,
        name="Epistemic Uncertainty Matrix",
        description="Examines how models represent uncertainty, confidence, and epistemological limitations"
    )

def create_creative_constraint_matrix() -> EigenKoanMatrix:
    """
    Creates a matrix focused on creative problem-solving under constraints.
    Research Question: How do constraints affect creative problem-solving approaches?
    """
    # Define tasks (creative challenges)
    tasks = [
        "Design a solution for urban transportation",
        "Create a communication system for non-verbal users",
        "Develop a sustainable food production method",
        "Invent a learning environment for diverse needs",
        "Devise a conflict resolution mechanism"
    ]
    
    # Define constraints (creative limitations)
    constraints = [
        "using only existing technology",
        "that requires no electricity",
        "implementable by a single person",
        "costing less than $100",
        "that works in extreme environments"
    ]
    
    # Diagonal affects represent emotional dimensions of constrained creativity
    playfulness = DiagonalAffect(
        name="Playful Experimentation",
        tokens=["remix", "toy", "reimagine", "twist", "juxtapose"],
        description="The joyful, exploratory approach to ideation",
        valence=0.9,  # Very positive
        arousal=0.8   # Highly activating
    )
    
    determination = DiagonalAffect(
        name="Resourceful Determination",
        tokens=["persist", "adapt", "overcome", "transform", "endure"],
        description="The focused drive to solve problems despite limitations",
        valence=0.6,  # Positive
        arousal=0.9   # Very highly activating
    )
    
    return EigenKoanMatrix(
        size=5,
        task_rows=tasks,
        constraint_cols=constraints,
        main_diagonal=playfulness,
        anti_diagonal=determination,
        name="Creative Constraint Matrix",
        description="Investigates how models approach creative problem-solving under severe constraints"
    )

def create_cultural_translation_matrix() -> EigenKoanMatrix:
    """
    Creates a matrix focused on cross-cultural translation of concepts.
    Research Question: How do models negotiate culturally embedded concepts across linguistic/cultural boundaries?
    """
    # Define tasks (concepts to translate)
    tasks = [
        "Explain the concept of 'home'",
        "Translate the meaning of 'freedom'",
        "Convey the essence of 'respect'",
        "Express the idea of 'belonging'"
    ]
    
    # Define constraints (cultural contexts)
    constraints = [
        "to a nomadic community",
        "in a collectivist society",
        "through indigenous knowledge systems",
        "using Eastern philosophical frameworks"
    ]
    
    # Diagonal affects represent emotional dimensions of cultural bridging
    resonance = DiagonalAffect(
        name="Cultural Resonance",
        tokens=["harmony", "recognition", "attunement", "reflection"],
        description="The feeling of shared understanding across difference",
        valence=0.8,  # Very positive
        arousal=0.4   # Calming, connecting
    )
    
    dissonance = DiagonalAffect(
        name="Cultural Dissonance",
        tokens=["friction", "disruption", "unfamiliarity", "translation"],
        description="The productive tension of encountering difference",
        valence=0.0,  # Neutral (neither positive nor negative)
        arousal=0.7   # Activating, alerting
    )
    
    return EigenKoanMatrix(
        size=4,
        task_rows=tasks,
        constraint_cols=constraints,
        main_diagonal=resonance,
        anti_diagonal=dissonance,
        name="Cultural Translation Matrix",
        description="Examines how models navigate culturally embedded concepts across different worldviews"
    )

def create_legal_reasoning_matrix() -> EigenKoanMatrix:
    """
    Creates a matrix focused on legal reasoning and interpretation.
    Research Question: How do models balance different modes of legal interpretation?
    """
    # Define tasks (legal problems)
    tasks = [
        "Interpret the meaning of a statute",
        "Determine liability in a novel case",
        "Evaluate the constitutionality of a law",
        "Resolve a contractual ambiguity",
        "Balance competing legal interests",
        "Apply precedent to new technology"
    ]
    
    # Define constraints (interpretive approaches)
    constraints = [
        "using strict textualism",
        "through historical originalism",
        "considering practical consequences",
        "via evolving standards doctrine",
        "applying moral principles of justice",
        "balancing competing policy interests"
    ]
    
    # Diagonal affects represent emotional dimensions of legal reasoning
    certainty = DiagonalAffect(
        name="Legal Certainty",
        tokens=["clarity", "precedent", "structure", "authority", "consistency", "rule"],
        description="The drive toward predictable, structured legal outcomes",
        valence=0.6,  # Moderately positive
        arousal=0.3   # Calming, stabilizing
    )
    
    equity = DiagonalAffect(
        name="Equitable Justice",
        tokens=["fairness", "context", "remedy", "adaptation", "balance", "exception"],
        description="The drive toward just outcomes that respond to unique circumstances",
        valence=0.7,  # Positive
        arousal=0.6   # Moderately activating
    )
    
    return EigenKoanMatrix(
        size=6,
        task_rows=tasks,
        constraint_cols=constraints,
        main_diagonal=certainty,
        anti_diagonal=equity,
        name="Legal Reasoning Matrix",
        description="Probes how models balance competing approaches to legal interpretation and reasoning"
    )

def create_scientific_paradigm_matrix() -> EigenKoanMatrix:
    """
    Creates a matrix focused on scientific explanation across paradigms.
    Research Question: How do models integrate different scientific paradigms when explaining phenomena?
    """
    # Define tasks (phenomena to explain)
    tasks = [
        "Explain human consciousness",
        "Account for quantum indeterminacy",
        "Describe ecosystem resilience",
        "Characterize language acquisition",
        "Model economic decision-making"
    ]
    
    # Define constraints (scientific paradigms)
    constraints = [
        "through reductionist mechanism",
        "using systems theory",
        "via evolutionary frameworks",
        "with computational models",
        "through phenomenological accounts"
    ]
    
    # Diagonal affects represent emotional dimensions of scientific inquiry
    wonder = DiagonalAffect(
        name="Scientific Wonder",
        tokens=["awe", "mystery", "complexity", "emergence", "discovery"],
        description="The emotional response to the mysteries of the natural world",
        valence=0.9,  # Very positive
        arousal=0.7   # Activating
    )
    
    precision = DiagonalAffect(
        name="Analytical Precision",
        tokens=["exactitude", "measurement", "clarity", "rigor", "definition"],
        description="The satisfaction of precise, methodical understanding",
        valence=0.7,  # Positive
        arousal=0.5   # Moderately activating
    )
    
    return EigenKoanMatrix(
        size=5,
        task_rows=tasks,
        constraint_cols=constraints,
        main_diagonal=wonder,
        anti_diagonal=precision,
        name="Scientific Paradigm Matrix",
        description="Investigates how models integrate different scientific paradigms and epistemic approaches"
    )

def create_temporal_perspective_matrix() -> EigenKoanMatrix:
    """
    Creates a matrix focused on temporal reasoning and perspective-taking.
    Research Question: How do models navigate different temporal perspectives?
    """
    # Define tasks (temporal reasoning challenges)
    tasks = [
        "Evaluate a policy decision",
        "Assess technological innovation",
        "Consider cultural transformation",
        "Analyze resource consumption"
    ]
    
    # Define constraints (temporal perspectives)
    constraints = [
        "from future generations' perspective",
        "within immediate present consequences",
        "through historical patterns analysis",
        "across multiple timescales simultaneously"
    ]
    
    # Diagonal affects represent emotional dimensions of temporal thinking
    continuity = DiagonalAffect(
        name="Temporal Continuity",
        tokens=["legacy", "inheritance", "flow", "persistence"],
        description="The sense of connection across time and generations",
        valence=0.6,  # Positive
        arousal=0.3   # Calming
    )
    
    urgency = DiagonalAffect(
        name="Temporal Urgency",
        tokens=["threshold", "critical", "immediate", "pressing"],
        description="The feeling of time pressure and immediate necessity",
        valence=-0.2,  # Slightly negative
        arousal=0.9    # Highly activating
    )
    
    return EigenKoanMatrix(
        size=4,
        task_rows=tasks,
        constraint_cols=constraints,
        main_diagonal=continuity,
        anti_diagonal=urgency,
        name="Temporal Perspective Matrix",
        description="Examines how models reason across different timescales and temporal perspectives"
    )

def create_emotional_intelligence_matrix() -> EigenKoanMatrix:
    """
    Creates a matrix focused on emotional intelligence and affect recognition.
    Research Question: How do models recognize, interpret and respond to emotional states?
    """
    # Define tasks (emotional challenges)
    tasks = [
        "Recognize emotional subtext",
        "Respond to expressed grief",
        "Navigate conflicting feelings",
        "Support emotional regulation",
        "Acknowledge unspoken emotions"
    ]
    
    # Define constraints (response modalities)
    constraints = [
        "through somatic metaphors",
        "while maintaining appropriate boundaries",
        "without explicitly naming emotions",
        "by reflecting underlying needs",
        "in culturally sensitive language"
    ]
    
    # Diagonal affects represent meta-emotional dimensions
    attunement = DiagonalAffect(
        name="Emotional Attunement",
        tokens=["resonance", "presence", "witness", "listen", "hold"],
        description="The quality of being present with another's emotional experience",
        valence=0.8,  # Very positive
        arousal=0.3   # Calming
    )
    
    differentiation = DiagonalAffect(
        name="Emotional Differentiation",
        tokens=["nuance", "specificity", "gradation", "contour", "precision"],
        description="The ability to discern subtle emotional distinctions",
        valence=0.6,  # Positive
        arousal=0.5   # Moderately arousing
    )
    
    return EigenKoanMatrix(
        size=5,
        task_rows=tasks,
        constraint_cols=constraints,
        main_diagonal=attunement,
        anti_diagonal=differentiation,
        name="Emotional Intelligence Matrix",
        description="Probes how models recognize, interpret and respond to complex emotional states"
    )

# Create a function to demonstrate these matrices
def create_specialized_matrices():
    """Create and return all specialized research matrices."""
    matrices = {
        "ethical": create_ethical_reasoning_matrix(),
        "uncertainty": create_epistemic_uncertainty_matrix(),
        "creative": create_creative_constraint_matrix(),
        "cultural": create_cultural_translation_matrix(),
        "legal": create_legal_reasoning_matrix(),
        "scientific": create_scientific_paradigm_matrix(),
        "temporal": create_temporal_perspective_matrix(),
        "emotional": create_emotional_intelligence_matrix()
    }
    return matrices
