import copy
import math
from multiprocessing import Manager, Pool, managers
from evaluadorBlosum import evaluadorBlosum
import numpy
import random
import concurrent.futures
import time

class bacteria():

    def __init__(self, numBacterias, gop=-10, gep=-1):
        manager = Manager()
        self.blosumScore = manager.list([0.0] * numBacterias) # Stores SPS Afín
        self.tablaAtract = manager.list([0.0] * numBacterias)
        self.tablaRepel = manager.list([0.0] * numBacterias)
        self.tablaInteraction = manager.list([0.0] * numBacterias)
        self.tablaFitness = manager.list([0.0] * numBacterias)
        self.GOP = gop
        self.GEP = gep
        # No shared evaluator instance

    def resetListas(self, numBacterias):
        for i in range(numBacterias):
            try:
                self.blosumScore[i] = 0.0; self.tablaAtract[i] = 0.0; self.tablaRepel[i] = 0.0
                self.tablaInteraction[i] = 0.0; self.tablaFitness[i] = 0.0
            except IndexError: pass
            except Exception: pass

    # --- Sum-of-Pairs Score with Affine Gap Penalty ---
    def calcular_sps_afin(self, alignment):
        try:
             evaluador = evaluadorBlosum(gop=self.GOP, gep=self.GEP)
        except Exception: return -float('inf')
        if not alignment or not isinstance(alignment, list) or not alignment[0] or not isinstance(alignment[0], list): return -float('inf')
        num_seqs = len(alignment)
        try: num_cols = len(alignment[0]);
        except IndexError: return 0.0
        if num_cols == 0: return 0.0
        total_substitution_score = 0.0; total_gap_penalty = 0.0
        # Substitution
        for j in range(num_cols):
            column = [alignment[i][j] if i < num_seqs and j < len(alignment[i]) else '-' for i in range(num_seqs)]
            for i1 in range(num_seqs):
                for i2 in range(i1 + 1, num_seqs):
                    char1 = column[i1]; char2 = column[i2]
                    if char1 != '-' and char2 != '-':
                        total_substitution_score += evaluador.get_blosum_score(char1, char2)
        # Gaps
        for i in range(num_seqs):
             in_gap = False
             if i >= len(alignment) or not isinstance(alignment[i], list): continue
             seq = alignment[i]
             for j in range(num_cols):
                 # Check bounds before accessing seq[j]
                 is_gap = (j < len(seq) and seq[j] == '-')
                 if is_gap:
                     if not in_gap: total_gap_penalty += self.GOP; in_gap = True
                     else: total_gap_penalty += self.GEP
                 else: in_gap = False
        return total_substitution_score + total_gap_penalty

    # --- Operadores de Modificación (Sin cambios) ---
    def _insert_gap(self, alignment, seq_idx):
        if seq_idx >= len(alignment) or not isinstance(alignment[seq_idx], list): return False
        seq_len = len(alignment[seq_idx]); pos = random.randint(0, seq_len)
        alignment[seq_idx].insert(pos, '-'); return True
    def _delete_gap(self, alignment, seq_idx):
        if seq_idx >= len(alignment) or not isinstance(alignment[seq_idx], list): return False
        gap_indices = [k for k, char in enumerate(alignment[seq_idx]) if char == '-']
        if gap_indices: pos_to_del = random.choice(gap_indices); del alignment[seq_idx][pos_to_del]; return True
        return False
    def _shift_gap(self, alignment, seq_idx):
        if seq_idx >= len(alignment) or not isinstance(alignment[seq_idx], list): return False
        seq_len = len(alignment[seq_idx])
        gap_indices = [k for k, char in enumerate(alignment[seq_idx]) if char == '-']
        if gap_indices:
            gap_idx = random.choice(gap_indices); direction = random.choice([-1, 1]); new_pos = gap_idx + direction
            if 0 <= new_pos < seq_len and alignment[seq_idx][new_pos] != '-':
                alignment[seq_idx][gap_idx], alignment[seq_idx][new_pos] = alignment[seq_idx][new_pos], alignment[seq_idx][gap_idx]
                return True
        return False
    def _mover_bloque_gaps(self, alignment, seq_idx):
        if seq_idx >= len(alignment) or not isinstance(alignment[seq_idx], list): return False
        seq = alignment[seq_idx]; seq_len = len(seq);
        if seq_len < 2: return False
        gap_blocks = []; start = -1
        for k, char in enumerate(seq):
            if char == '-' and start == -1: start = k
            elif char != '-' and start != -1: gap_blocks.append((start, k, k - start)); start = -1
        if start != -1: gap_blocks.append((start, seq_len, seq_len - start))
        if not gap_blocks: return False
        block_start, block_end, block_len = random.choice(gap_blocks)
        possible_new_starts = [p for p in range(seq_len - block_len + 1) if p < block_start or p >= block_end]
        if not possible_new_starts: return False
        new_start = random.choice(possible_new_starts); gap_block_content = ['-'] * block_len
        remaining_seq = seq[:block_start] + seq[block_end:]
        new_seq = remaining_seq[:new_start] + gap_block_content + remaining_seq[new_start:]
        alignment[seq_idx][:] = new_seq; return True
    def _mover_bloque_residuos(self, alignment, seq_idx, block_size=10):
        if seq_idx >= len(alignment) or not isinstance(alignment[seq_idx], list): return False
        seq = alignment[seq_idx]; seq_len = len(seq)
        if seq_len <= block_size: return False
        block_start = random.randint(0, seq_len - block_size); block_end = block_start + block_size
        possible_new_starts = [p for p in range(seq_len - block_size + 1) if abs(p - block_start) > block_size // 2]
        if not possible_new_starts: possible_new_starts = [p for p in range(seq_len - block_size + 1) if p != block_start]
        if not possible_new_starts: return False
        new_start = random.choice(possible_new_starts); block_content = seq[block_start:block_end]
        remaining_seq = seq[:block_start] + seq[block_end:]
        new_seq = remaining_seq[:new_start] + block_content + remaining_seq[new_start:]
        alignment[seq_idx][:] = new_seq; return True

    # --- Chemotaxis Step: Modifies alignment, DOES NOT update self.blosumScore ---
    def paso_quimiotactico(self, poblacion, nado_steps):
        num_bacterias = len(poblacion)
        operators = [self._insert_gap, self._delete_gap, self._shift_gap, self._mover_bloque_gaps, self._mover_bloque_residuos]

        # Get initial scores ONLY for comparison within the loop
        initial_local_scores = {}
        for i in range(num_bacterias):
             try: initial_local_scores[i] = self.calcular_sps_afin(poblacion[i])
             except Exception: initial_local_scores[i] = -float('inf')

        for i in range(num_bacterias):
            if initial_local_scores[i] == -float('inf'): continue

            current_alignment_ref = poblacion[i] # Reference to managed list item
            current_score = initial_local_scores[i] # Local score for comparison

            for step in range(nado_steps):
                # Try modification on a copy
                temp_alignment = copy.deepcopy(current_alignment_ref)
                operator_func = random.choice(operators)
                num_seqs = len(temp_alignment)
                if num_seqs == 0: break
                # Ensure seq_idx is valid for the current state of temp_alignment
                if num_seqs > 0:
                     seq_idx = random.randint(0, num_seqs - 1)
                else: continue # Skip if somehow num_seqs became 0

                modified = False
                try:
                    if operator_func == self._mover_bloque_residuos: modified = operator_func(temp_alignment, seq_idx, block_size=random.randint(5,15))
                    else: modified = operator_func(temp_alignment, seq_idx)
                except Exception: modified = False

                if modified:
                    self.cuadra_individual(temp_alignment)
                    modified_score = self.calcular_sps_afin(temp_alignment)

                    # Accept if >= and valid score
                    if isinstance(modified_score, (int, float)) and numpy.isfinite(modified_score) and modified_score >= current_score:
                        # --- Update the actual alignment in the population list ---
                        poblacion[i][:] = temp_alignment
                        current_score = modified_score # Update local score for next step comparison
                        # --- DO NOT update self.blosumScore here ---
                        current_alignment_ref = poblacion[i] # Update local reference

    # --- Recalculate all SPS scores ---
    def recalcular_scores_sps_poblacion(self, poblacion):
         """Recalculates SPS Afín score for all bacteria and updates self.blosumScore."""
         print(f"[{time.strftime('%H:%M:%S')}] Recalculando scores SPS...") # Timestamp for clarity
         num_bacterias = len(poblacion)
         for i in range(num_bacterias):
              try:
                   current_sps = self.calcular_sps_afin(poblacion[i])
                   # Ensure score is valid before assigning
                   if isinstance(current_sps, (int, float)) and numpy.isfinite(current_sps):
                        self.blosumScore[i] = current_sps
                   else:
                        self.blosumScore[i] = -float('inf') # Assign bad score if calculation failed
              except Exception as e:
                   print(f"Error recalculando SPS bacteria {i}: {e}")
                   self.blosumScore[i] = -float('inf')

    # --- Cuadra methods (sin cambios) ---
    def cuadra_individual(self, alignment):
         if not isinstance(alignment, list) or not alignment: return
         maxLen = max((len(s) for s in alignment if isinstance(s, list)), default=0)
         for t in range(len(alignment)):
             if isinstance(alignment[t], list):
                 gap_count = maxLen - len(alignment[t])
                 if gap_count > 0: alignment[t].extend(["-"] * gap_count)
    def cuadra(self, poblacion):
        num_bacterias = len(poblacion)
        for i in range(num_bacterias):
            try: self.cuadra_individual(poblacion[i])
            except Exception as e: print(f"Error Cuadra {i}: {e}")

    # --- Interaction Calculation (sin cambios) ---
    def _calculate_single_interaction_wrapper(self, args):
        indexBacteria, normalized_scores, dAttr, wAttr, dRepel, wRepel = args
        my_norm_score = 0.0; attr_sum = 0.0; repel_sum = 0.0; nfe_count = 0
        try:
            if 0 <= indexBacteria < len(normalized_scores) and normalized_scores[indexBacteria] is not None and numpy.isfinite(normalized_scores[indexBacteria]):
                my_norm_score = normalized_scores[indexBacteria]
            for other_norm_score in normalized_scores:
                if other_norm_score is None or not numpy.isfinite(other_norm_score): other_norm_score = 0.0
                try:
                    diff_sq = (my_norm_score - other_norm_score) ** 2.0
                    attr_term = dAttr * numpy.exp(-wAttr * diff_sq)
                    repel_term = dRepel * numpy.exp(-wRepel * diff_sq)
                    if numpy.isfinite(attr_term): attr_sum += attr_term
                    if numpy.isfinite(repel_term): repel_sum += repel_term
                    nfe_count += 2
                except Exception: nfe_count += 2
        except Exception: return indexBacteria, 0.0, 0.0, nfe_count
        return indexBacteria, attr_sum, repel_sum, nfe_count
    def creaTablasAtractRepel(self, poblacion, dAttr, wAttr, dRepel, wRepel):
        # --- Uses self.blosumScore which should be up-to-date after recalculate step ---
        num_bacterias = len(poblacion); total_nfe_increment = 0
        if num_bacterias == 0: return 0
        try:
            current_sps_scores = list(self.blosumScore) # Use the recently recalculated scores
            if len(current_sps_scores) != num_bacterias: raise ValueError("SPS score list size mismatch.")
            finite_scores = [s for s in current_sps_scores if isinstance(s, (int, float)) and numpy.isfinite(s)]
            if not finite_scores: normalized_scores = [0.0] * num_bacterias
            else:
                 min_score = min(finite_scores); max_score = max(finite_scores); score_range = max_score - min_score
                 if score_range > 1e-9: normalized_scores = [(s - min_score) / score_range if isinstance(s, (int, float)) and numpy.isfinite(s) else None for s in current_sps_scores]
                 else: normalized_scores = [0.5 if isinstance(s, (int, float)) and numpy.isfinite(s) else None for s in current_sps_scores]
            args_list = [(i, normalized_scores, dAttr, wAttr, dRepel, wRepel) for i in range(num_bacterias)]
            results = []; pool = None
            try: pool = Pool(); results = pool.map(self._calculate_single_interaction_wrapper, args_list)
            finally:
                if pool: pool.close(); pool.join()
            temp_nfe = [0] * num_bacterias; processed_indices = set()
            for result_tuple in results:
                try:
                     if isinstance(result_tuple, tuple) and len(result_tuple) == 4:
                         index, attr, repel, nfe = result_tuple
                         if 0 <= index < num_bacterias and index not in processed_indices:
                             self.tablaAtract[index] = attr if numpy.isfinite(attr) else 0.0
                             self.tablaRepel[index] = repel if numpy.isfinite(repel) else 0.0
                             temp_nfe[index] = nfe; processed_indices.add(index)
                except Exception: pass
            total_nfe_increment = sum(temp_nfe)
        except Exception as e: print(f"Error crítico en creaTablasAtractRepel: {e}"); return 0
        return total_nfe_increment

    # --- Final Calculations & Selection (sin cambios) ---
    def creaTablaInteraction(self):
        num_bacterias = len(self.tablaAtract)
        for i in range(num_bacterias):
            try:
                attr_val = self.tablaAtract[i] if isinstance(self.tablaAtract[i], (int, float)) and numpy.isfinite(self.tablaAtract[i]) else 0.0
                repel_val = self.tablaRepel[i] if isinstance(self.tablaRepel[i], (int, float)) and numpy.isfinite(self.tablaRepel[i]) else 0.0
                self.tablaInteraction[i] = attr_val + repel_val
            except IndexError: pass
            except Exception: 
                if i < len(self.tablaInteraction): self.tablaInteraction[i] = 0.0
    def creaTablaFitness(self):
        num_bacterias = len(self.tablaInteraction)
        for i in range(num_bacterias):
            try:
                valorSPS = self.blosumScore[i] if isinstance(self.blosumScore[i], (int, float)) and numpy.isfinite(self.blosumScore[i]) else -float('inf')
                valorInteract = self.tablaInteraction[i] if isinstance(self.tablaInteraction[i], (int, float)) and numpy.isfinite(self.tablaInteraction[i]) else 0.0
                if valorSPS == -float('inf'): self.tablaFitness[i] = -float('inf')
                else: self.tablaFitness[i] = valorSPS + valorInteract
            except IndexError: pass
            except Exception: 
                if i < len(self.tablaFitness): self.tablaFitness[i] = -float('inf')
    def obtieneBest(self, globalNFE_actual):
        # --- Uses self.tablaFitness which depends on the recalculated self.blosumScore ---
        bestIdx = -1; bestFitness = -float('inf')
        try:
            fitness_list = list(self.tablaFitness)
            num_bacterias = len(fitness_list)
            if num_bacterias == 0: return -1, -float('inf')
            valid_indices = [idx for idx, fit in enumerate(fitness_list) if isinstance(fit, (int, float)) and numpy.isfinite(fit)]
            if not valid_indices: print("Advertencia: No hay fitness válidos en obtieneBest."); return -1, -float('inf')
            bestIdx = max(valid_indices, key=lambda idx: fitness_list[idx])
            bestFitness = fitness_list[bestIdx]
            bestSPS_str = 'N/A'
            # --- Read the score directly from the up-to-date self.blosumScore ---
            if 0 <= bestIdx < len(self.blosumScore) and isinstance(self.blosumScore[bestIdx], (int, float)): bestSPS_str = f"{self.blosumScore[bestIdx]:.1f}"
            bestInteract_str = 'N/A'
            if 0 <= bestIdx < len(self.tablaInteraction) and isinstance(self.tablaInteraction[bestIdx], (int, float)): bestInteract_str = f"{self.tablaInteraction[bestIdx]:.4f}"
            print(f"--- Mejor Iteración --- Índice: {bestIdx}, Fitness: {bestFitness:.4f}, SPS_Afín: {bestSPS_str}, Interacción: {bestInteract_str}, NFE Total: {globalNFE_actual}")
        except Exception as e: print(f"Error obtieneBest: {e}"); return -1, -float('inf')
        return bestIdx, bestFitness
    def replaceWorst(self, poblacion, best_global_idx):
        worstIdx = -1; worstFitness = float('inf')
        num_bacterias = len(poblacion)
        if num_bacterias <= 1 or best_global_idx is None or not (0 <= best_global_idx < num_bacterias): return
        try:
            fitness_list = list(self.tablaFitness)
            if len(fitness_list) != num_bacterias: return
            valid_indices = [idx for idx, fit in enumerate(fitness_list) if isinstance(fit, (int, float)) and numpy.isfinite(fit)]
            if not valid_indices: return
            worstIdx = min(valid_indices, key=lambda idx: fitness_list[idx])
            worstFitness = fitness_list[worstIdx]
            if worstIdx != -1 and worstIdx != best_global_idx:
                try:
                    # Get data from the known best (already copied to veryBest[2])
                    # Or should we copy directly from population[best_global_idx]? Let's try direct.
                    managed_list_item = poblacion[best_global_idx]
                    best_alignment_list = [list(seq) for seq in managed_list_item]
                    copied_data = copy.deepcopy(best_alignment_list)
                    poblacion[worstIdx] = copied_data # Assign copy to worst
                except Exception as e: print(f"Error deepcopy/asignación replaceWorst: {e}")
        except Exception as e: print(f"Error inesperado replaceWorst: {e}")