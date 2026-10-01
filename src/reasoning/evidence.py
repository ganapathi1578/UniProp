import re
from typing import Dict, Any, Optional, List
from src.normalization.scene_graph_normalizer import NormalizedSceneGraph

class EvaluatorEvidence:
    """Evaluates candidate propositions against SG evidence using the AGQA program."""
    
    def __init__(self, sg: NormalizedSceneGraph):
        self.sg = sg

    def evaluate_temporal(self, candidate: str, program: str) -> str:
        # Example program: Compare([before, after], Exists(window, Iterate(Localize(temporal tag, making some food), Filter(frame, [objects]))))
        # Or: Compare([before, after], Query(action, Iterate(Localize(temporal tag, making some food), Filter(video, [actions]))))
        
        # We need to find the reference action:
        # Localize(..., <action>)
        loc_match = re.search(r'Localize\([^,]+,\s*([^)]+)\)', program)
        if not loc_match:
            return "UNKNOWN"
            
        ref_action = loc_match.group(1).strip()
        
        # Find when ref_action happens
        ref_starts, ref_ends = [], []
        for act in self.sg.actions.values():
            if act.phrase == ref_action:
                ref_starts.append(act.start_secs)
                ref_ends.append(act.end_secs)
                
        if not ref_starts:
            return "UNKNOWN"
            
        ref_min = min(ref_starts)
        ref_max = max(ref_ends)
        
        # Are we looking for an object or an action?
        if "Exists(" in program or "Query(object" in program:
            # Find the target object
            # Exists(window, ...)
            obj_match = re.search(r'Exists\(([^,]+),', program)
            if not obj_match:
                return "UNKNOWN"
            target_obj = obj_match.group(1).strip()
            
            # Check frames
            found_before = False
            found_after = False
            for f in self.sg.frames.values():
                for obj in f.objects.values():
                    if obj.name == target_obj:
                        if f.secs < ref_min:
                            found_before = True
                        if f.secs > ref_max:
                            found_after = True
                            
            if candidate == "before" and found_before: return "TRUE"
            if candidate == "after" and found_after: return "TRUE"
            return "FALSE"
            
        elif "Query(action" in program or "Exists(action" in program:
            # Find target action
            act_match = re.search(r'Exists\(([^,]+),', program)
            if not act_match:
                # If no explicit Exists, maybe it's just checking if candidate action happened before/after
                target_act = candidate
            else:
                target_act = act_match.group(1).strip()
                
            found_before = False
            found_after = False
            for act in self.sg.actions.values():
                if act.phrase == target_act:
                    if act.end_secs <= ref_min:
                        found_before = True
                    if act.start_secs >= ref_max:
                        found_after = True
                        
            if candidate == "before" and found_before: return "TRUE"
            if candidate == "after" and found_after: return "TRUE"
            return "FALSE"

        return "UNKNOWN"

    def evaluate_object(self, candidate: str, program: str) -> str:
        # Example: Query(object, Iterate(Localize(after, making some food), Filter(frame, [objects])))
        
        loc_match = re.search(r'Localize\(([^,]+),\s*([^)]+)\)', program)
        if not loc_match:
            # Maybe it's checking what they did first/last?
            # Or just interacting with?
            # "Exists(window, Filter(video, [objects]))"
            obj_match = re.search(r'Exists\(([^,]+),', program)
            if obj_match:
                return "TRUE" if candidate == obj_match.group(1).strip() else "FALSE"
                
            return "UNKNOWN"
            
        rel = loc_match.group(1).strip()
        ref_action = loc_match.group(2).strip()
        
        ref_starts, ref_ends = [], []
        for act in self.sg.actions.values():
            if act.phrase == ref_action:
                ref_starts.append(act.start_secs)
                ref_ends.append(act.end_secs)
                
        if not ref_starts:
            return "UNKNOWN"
            
        ref_min = min(ref_starts)
        ref_max = max(ref_ends)
        
        # Check frames based on rel
        for f in self.sg.frames.values():
            if rel == "after" and f.secs > ref_max:
                for obj in f.objects.values():
                    if obj.name == candidate:
                        return "TRUE"
            elif rel == "before" and f.secs < ref_min:
                for obj in f.objects.values():
                    if obj.name == candidate:
                        return "TRUE"
            elif rel == "during" and ref_min <= f.secs <= ref_max:
                for obj in f.objects.values():
                    if obj.name == candidate:
                        return "TRUE"
                        
        return "FALSE"

    def evaluate_binary(self, candidate: str, program: str, q_text: str) -> str:
        # Example: Verify(Exists(window, Filter(video, [objects])))
        # candidate is "Yes" or "No"
        
        obj_match = re.search(r'Exists\(([^,]+),', program)
        if obj_match:
            target = obj_match.group(1).strip()
            
            # Is target in SG objects or actions?
            found = False
            for f in self.sg.frames.values():
                for obj in f.objects.values():
                    if obj.name == target:
                        found = True
                        break
            if not found:
                for act in self.sg.actions.values():
                    if target in act.phrase:
                        found = True
                        break
                        
            is_true = found
            if candidate.lower() == "yes":
                return "TRUE" if is_true else "FALSE"
            else:
                return "TRUE" if not is_true else "FALSE"
                
        # What if it's Verify(Localize(...))?
        # Very complex programs, let's parse the query text if program fails.
        # e.g., "did they interact with a window?"
        if "interact" in q_text.lower() or "with a " in q_text.lower():
            # simple heuristic for binary object query
            target = q_text.split("with a ")[-1].replace("?", "").strip()
            found = False
            for f in self.sg.frames.values():
                for obj in f.objects.values():
                    if target in obj.name:
                        found = True
            
            is_true = found
            if candidate.lower() == "yes":
                return "TRUE" if is_true else "FALSE"
            else:
                return "TRUE" if not is_true else "FALSE"
                
        return "UNKNOWN"

import pickle
from src.normalization.scene_graph_normalizer import normalize_video_sg

class TruthEvaluator:
    def __init__(self, scenegraph_path: str):
        print(f"Loading scenegraph evidence from {scenegraph_path}...")
        with open(scenegraph_path, 'rb') as f:
            self.raw_data = pickle.load(f)
        self.sg_cache: Dict[str, NormalizedSceneGraph] = {}

    def get_scenegraph(self, video_id: str) -> Optional[NormalizedSceneGraph]:
        if video_id not in self.raw_data:
            return None
            
        if video_id not in self.sg_cache:
            self.sg_cache[video_id] = normalize_video_sg(video_id, self.raw_data[video_id])
            
        return self.sg_cache[video_id]

    def evaluate(self, normalized_question: Any, candidate: str, evidence_context: Any) -> str:
        sg = self.get_scenegraph(normalized_question.video_id)
        if not sg:
            return "MISSING_EVIDENCE"
            
        evaluator = EvaluatorEvidence(sg)
        
        domain = normalized_question.answer_type
        prog = normalized_question.source_program or normalized_question.program
        
        if domain == "temporal":
            return evaluator.evaluate_temporal(candidate, prog)
        elif domain == "binary":
            return evaluator.evaluate_binary(candidate, prog, normalized_question.query)
        elif domain == "object":
            return evaluator.evaluate_object(candidate, prog)
            
        return "UNKNOWN"
