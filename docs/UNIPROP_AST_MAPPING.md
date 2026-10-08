# UniProp AST-to-Proposition Mapping

This document formalizes the mapping from AGQA root operations to their semantic domains, reasoning modes, and proposition templates. This ensures the proposition verbalizer mathematically preserves the AST semantics without falling back to arbitrary string concatenation.

## 1. Architectural Layers

### Layer 1: Semantic Domain (What is returned)
*   **BOOLEAN**: The result is a truth value (e.g., `Exists`, `And`).
*   **OBJECT**: The result is an object/entity (e.g., `Query(class)`).
*   **ACTION**: The result is an action/event (e.g., `Query(action)`).
*   **RELATION**: The result is a spatial or contact relationship (e.g., `Query(relation)`).

### Layer 2: Reasoning Mode (How it is derived)
*   **DIRECT**: Simple lookup or attribute retrieval.
*   **TEMPORAL**: Involves a temporal window (`before`, `after`, `while`).
*   **COMPARISON**: Evaluates a greater/lesser relationship between two items.
*   **SUPERLATIVE**: Evaluates extreme bounds (`longest`, `shortest`, `first`, `last`).
*   **LOGICAL**: Boolean composition (`AND`, `XOR`, `NOT`).

---

## 2. AGQA Root AST Mapping Table

| AGQA Root Operation | Semantic Domain | Reasoning Mode | Conceptual Proposition Template |
| :--- | :--- | :--- | :--- |
| `Exists(target, collection)` | **BOOLEAN** | DIRECT / TEMPORAL | `[Polarity] the person {interacted with/did} [target] [TemporalContext(collection)]` |
| `Verify(target, collection)` | **BOOLEAN** | DIRECT | `[Polarity] the [target] was [Relation/Action] [TemporalContext]` |
| `Query(class, collection)` | **OBJECT** | DIRECT / TEMPORAL | `The person [ActionContext] [CandidateObject] [TemporalContext]` |
| `Query(action, collection)` | **ACTION** | DIRECT / TEMPORAL | `The person [CandidateAction] [TemporalContext]` |
| `Query(relation, collection)` | **RELATION** | DIRECT | `The person was [CandidateRelation] the [ObjectContext] [TemporalContext]` |
| `Choose(opt1, opt2, collection)` | **OBJECT** / **ACTION** | LOGICAL | *(Uses same template as `Query`, but restricts generation to `opt1`/`opt2` candidates)* |
| `Compare(opt1, opt2, condition)` | **OBJECT** / **ACTION** | COMPARISON | `[Candidate] [ComparisonPredicate] than [OtherCandidate] [TemporalContext]` |
| `Superlative(attribute, collection)` | **OBJECT** / **ACTION** | SUPERLATIVE | `The person [Action/Relation] [Candidate] for the [SuperlativeModifier] [TemporalContext]` |
| `And(stmt1, stmt2)` | **BOOLEAN** | LOGICAL | `[Prop(stmt1)] and [Prop(stmt2)]` |
| `Xor(stmt1, stmt2)` | **BOOLEAN** | LOGICAL | `Either [Prop(stmt1)] or [Prop(stmt2)], but not both.` |

---

## 3. Proposition Template Structure

To prevent scope ambiguity and ensure proper generation, the verbalizer will construct sentences using a strict slot-filling model:

```python
@dataclass
class PropositionTemplate:
    subject: str = "The person"
    predicate: str       # e.g., "interacted with", "was", "did"
    arguments: List[str] # e.g., ["a window"], ["standing up"]
    temporal_constraints: Optional[str] # e.g., "after making some food"
    comparison_constraints: Optional[str] # e.g., "for a longer amount of time"
    polarity: bool = True # True = Positive, False = Negative (allows "It is false that...")
```

### Negation Handling (The Polarity Rule)
To solve the scope issue with complex logic like `"did not put down a blanket but not a broom"`, `BOOLEAN` evaluations that result in `FALSE` will use explicit wrapper negation when nested, or strict verb negation for simple statements:
*   **Simple Negation**: `The person did not put down a blanket.`
*   **Complex/Compound Negation**: `It is false that [the person put down a blanket and did not put down a broom].` 

This completely removes logical ambiguity for downstream models.
