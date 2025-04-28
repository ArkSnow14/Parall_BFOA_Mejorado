import blosum as bl
import collections

class evaluadorBlosum():

    def __init__(self, gop=-10, gep=-1): # Accept GOP/GEP during initialization
        """Initializes the evaluator with Blosum matrix and gap penalties."""
        try:
            # Load the Blosum matrix object
            self._blosum_obj = bl.BLOSUM(62)
        except Exception as e:
            print(f"ERROR: Failed to initialize BLOSUM matrix: {e}")
            self._blosum_obj = None

        # Store Gap Penalties
        # Common values for Blosum62, adjust as needed
        self.GOP = gop # Gap Open Penalty (negative value)
        self.GEP = gep # Gap Extension Penalty (negative value, usually smaller magnitude than GOP)

        # Penalty for unknown characters (can be same as GEP or specific)
        self.unknown_penalty = self.GEP # Penalize unknown like extending a gap

    def get_blosum_score(self, A, B):
        """Gets the substitution score between two residues A and B."""
        # Default score if lookup fails
        default_score = self.unknown_penalty

        if self._blosum_obj is None:
             return default_score

        try:
            A_str = str(A).upper()
            B_str = str(B).upper()

            # If either is a gap, this function shouldn't be called by SPS calc,
            # but handle defensively. Return 0 as gaps are handled separately.
            if A_str == "-" or B_str == "-":
                return 0.0 # Gaps handled by GOP/GEP

            # Use .get() for safer dictionary-like access
            score_value = self._blosum_obj.get((A_str, B_str))

            if score_value is not None and isinstance(score_value, (int, float)):
                return float(score_value)
            else:
                # Character not found (e.g., 'X'), apply unknown penalty
                return float(self.unknown_penalty)

        except Exception as e:
            # print(f"Error in get_blosum_score for ({A}, {B}): {e}") # Optional debug
            return float(default_score)

    # getScore method remains for backward compatibility or simple use cases if needed,
    # but SPS calculation should use get_blosum_score and handle gaps explicitly.
    def getScore(self, A, B):
         # Simplified version using fixed penalties (less flexible than SPS Afín)
         if self._blosum_obj is None: return 0.0
         A_str = str(A).upper(); B_str = str(B).upper()
         if A_str == "-" or B_str == "-": return float(self.GEP) # Simple gap penalty (use GEP for consistency?)
         score = self._blosum_obj.get((A_str, B_str))
         if score is None: return float(self.unknown_penalty)
         return float(score) if isinstance(score, (int, float)) else float(self.unknown_penalty)