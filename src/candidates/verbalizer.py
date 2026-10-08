import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class PropositionTemplate:
    subject: str = "The person"
    predicate: str = ""
    arguments: List[str] = field(default_factory=list)
    temporal_constraints: Optional[str] = None
    comparison_constraints: Optional[str] = None
    polarity: bool = True

class PropositionVerbalizer:
    """
    Deterministically converts an AnswerSpecification and a candidate value 
    into a natural-language declarative proposition via a slot-filling template
    derived from the AST semantics.
    """
    def __init__(self):
        pass
        
    def _conjugate(self, verb: str) -> str:
        if verb.endswith("e"): return verb + "d"
        elif verb.endswith("y") and verb not in ["play", "stay"]: return verb[:-1] + "ied"
        else: return verb + "ed"

    def extract_from_ast(self, ast: Any, query: str = "") -> PropositionTemplate:
        """
        Derive PropositionTemplate using AST structural semantics as requested.
        Instead of regexing the question string, this inspects the AST node type.
        """
        tpl = PropositionTemplate()
        
        if not ast or not isinstance(ast, (list, tuple)):
            tpl.subject = "The answer to"
            tpl.predicate = f"'{query}' is"
            tpl.arguments = ["{candidate}"]
            return tpl
            
        op = ast[0]
        
        # BOOLEAN - Exists / Verify / And / Xor
        if op in ["Exists", "Verify"]:
            target = str(ast[1]).strip()
            # Try to determine if it's an action or object from query fallback if AST is too abstracted
            # But normally we'd pull from Filter(...)
            tpl.predicate = "interacted with" if op == "Exists" else "was"
            tpl.arguments = [target]
            return tpl
            
        # OBJECT / ACTION - Query
        if op == "Query":
            attr = str(ast[1]).lower()
            if attr == "action":
                tpl.predicate = "was"
                tpl.arguments = ["{candidate}"]
            elif attr in ["class", "object"]:
                tpl.predicate = "interacted with"
                tpl.arguments = ["{candidate}"]
            else:
                tpl.predicate = "was"
                tpl.arguments = ["{candidate}"]
            return tpl
            
        # COMPARISON - Compare
        if op == "Compare":
            # Check operands
            opts = ast[1]
            if isinstance(opts, list) and len(opts) >= 2:
                c1 = str(opts[0]).lower().strip()
                if c1 in ["longer", "shorter", "more", "less"]:
                    tpl.subject = "{candidate}"
                    tpl.predicate = "takes"
                    tpl.arguments = ["less time" if c1 == "shorter" else "more time"]
                elif c1 in ["before", "after"]:
                    tpl.predicate = "was something they did"
                    tpl.arguments = ["{candidate}"]
                else:
                    tpl.predicate = "was"
                    tpl.arguments = ["{candidate}"]
            return tpl
            
        # SUPERLATIVE - Superlative
        if op == "Superlative":
            mode = str(ast[1]).lower()
            tpl.subject = "{candidate}"
            tpl.predicate = "takes"
            tpl.arguments = ["less time" if mode == "min" else "more time"]
            return tpl
            
        # SELECTION - Choose
        if op == "Choose":
            tpl.predicate = "was"
            tpl.arguments = ["{candidate}"]
            return tpl
            
        # Default Fallback (could fall back to simple string embedding)
        tpl.subject = "The answer to"
        tpl.predicate = f"'{query}' is"
        tpl.arguments = ["{candidate}"]
        return tpl

    def verbalize(self, ast: Any, query: str, answer_kind: str, candidate: str, verbalization_info: Dict[str, Any] = None) -> str:
        """
        Produce a declarative proposition for the candidate using the strict AST template.
        """
        c = str(candidate).lower().strip()
        
        # Derive template strictly from AST semantics
        tpl = self.extract_from_ast(ast, query)
        
        # Determine polarity
        polarity = True
        if answer_kind == "boolean":
            if c in ("no", "false"):
                polarity = False
            c = ""
            
        # Format arguments
        args = []
        for arg in tpl.arguments:
            if "{candidate}" in arg:
                if c: 
                    args.append(arg.format(candidate=c))
            else:
                args.append(arg)
                
        # Resolve subject
        subj = tpl.subject
        if "{candidate}" in subj:
            subj = subj.format(candidate=c.capitalize())
            
        # Construct basic sentence
        if polarity:
            sent = f"{subj} {tpl.predicate} " + " ".join(args)
        else:
            sent = f"It is false that {subj.lower()} {tpl.predicate} " + " ".join(args)
            
        if tpl.temporal_constraints:
            sent += f" {tpl.temporal_constraints}"
            
        if tpl.comparison_constraints:
            sent += f" {tpl.comparison_constraints}"
            
        sent = re.sub(r'\s+', ' ', sent).strip()
        
        if not sent.endswith("."):
            sent += "."
            
        sent = sent[0].upper() + sent[1:]
            
        return sent
