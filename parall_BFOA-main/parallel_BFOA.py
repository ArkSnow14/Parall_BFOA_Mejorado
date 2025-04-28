import copy
from multiprocessing import Manager, Pool
import time
from bacteria import bacteria
import numpy
import csv
import random

from fastaReader import fastaReader
from evaluadorBlosum import evaluadorBlosum

if __name__ == "__main__":
    # --- Parameters ---
    numeroDeBacterias = 10
    iteraciones = 50
    nado_steps = 10
    GOP = -10.0; GEP = -1.0
    dAttr = 1.0; wAttr = 15.0
    dRepel = 0.5; wRepel = 20.0
    N_ed = 10; num_dispersar = 2

    # --- Report File Setup ---
    report_filename = "reporte_bacterias_SPS_ED_v4_recalc.csv" # New name
    report_file = None; csv_writer = None
    try:
        report_file = open(report_filename, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(report_file)
        header = ["Iteration", "BestIndex_Iter", "Fitness_Iter", "SPS_Afin_Score_Iter", "Interaction_Iter", "CumulativeNFE", "TimeSeconds_Iter"]
        csv_writer.writerow(header)
        print(f"Archivo de reporte '{report_filename}' abierto.")
    except IOError as e: print(f"Error abriendo reporte '{report_filename}': {e}"); report_file = None; csv_writer = None

    # --- Sequence Loading ---
    print("Leyendo secuencias FASTA...")
    try:
        reader = fastaReader(); secuencias_orig = reader.seqs; names_orig = reader.names
        if not secuencias_orig: raise ValueError("No sequences loaded.")
        print(f"Se cargaron {len(secuencias_orig)} secuencias.")
    except Exception as e: print(f"Error leyendo FASTA: {e}"); exit()
    numSec = len(secuencias_orig)

    # --- Preprocessing & Init ---
    try: secuencias_listas = [list(s) for s in secuencias_orig]
    except TypeError: print("Error: Sequences not iterable."); exit()
    print("Configurando Manager...");
    try: manager = Manager(); poblacion = manager.list([manager.list() for _ in range(numeroDeBacterias)]); names = manager.list(names_orig); globalNFE = 0
    except Exception as e: print(f"Error configurando Manager: {e}"); exit()
    def poblacionInicial():
        print("Creando población inicial (alineada por longitud)...")
        # ... (sin cambios en la lógica interna) ...
        try:
            max_len = max((len(s) for s in secuencias_listas), default=0)
            initial_alignment = []
            for seq in secuencias_listas:
                aligned_seq = list(seq); gap_count = max_len - len(aligned_seq)
                if gap_count > 0: aligned_seq.extend(['-'] * gap_count)
                initial_alignment.append(aligned_seq)
            for i in range(numeroDeBacterias): poblacion[i] = copy.deepcopy(initial_alignment)
            print(f"Población inicial creada con longitud {max_len}.")
            return True
        except Exception as e: print(f"Error en inicialización: {e}"); return False
    print("Inicializando operador bacterial...");
    try: operadorBacterial = bacteria(numeroDeBacterias, gop=GOP, gep=GEP)
    except Exception as e: print(f"Error init bacteria: {e}"); exit()

    # --- Tracking & Loop ---
    veryBest = [None, None, None]
    print("Iniciando Bucle BFOA (SPS Afín, Nuevos Ops, ED, Recalc)...")
    start_time = time.time()
    if not poblacionInicial(): exit()

    # --- Main BFOA Loop ---
    for it in range(iteraciones):
        iter_start_time = time.time()
        print(f"\n--- Iniciando Iteración {it + 1}/{iteraciones} ---")

        # 1. Chemotaxis (Modifies alignments only)
        print(f"[{it+1}] Ejecutando Pasos Quimiotácticos ({nado_steps} pasos)...")
        try: operadorBacterial.paso_quimiotactico(poblacion, nado_steps)
        except Exception as e: print(f"Error Paso Quimiotáctico: {e}"); continue

        # 2. Cuadra
        try: operadorBacterial.cuadra(poblacion)
        except Exception as e: print(f"Error Cuadra: {e}"); continue

        # --- 3. RECALCULAR SCORES SPS ANTES DE INTERACCIÓN ---
        try:
             operadorBacterial.recalcular_scores_sps_poblacion(poblacion)
        except Exception as e: print(f"Error recalculando Scores SPS: {e}"); continue

        # 4. Evaluate Interactions & Update NFE (Uses the recalculated scores)
        nfe_increment = 0
        try:
            nfe_increment = operadorBacterial.creaTablasAtractRepel(poblacion, dAttr, wAttr, dRepel, wRepel)
            globalNFE += nfe_increment
        except Exception as e: print(f"Error Interacciones: {e}"); continue

        # 5. Calculate Interaction Table
        try: operadorBacterial.creaTablaInteraction()
        except Exception as e: print(f"Error Tabla Interacción: {e}"); continue

        # 6. Calculate Fitness Table (Uses recalculated SPS + interaction)
        try: operadorBacterial.creaTablaFitness()
        except Exception as e: print(f"Error Tabla Fitness: {e}"); continue

        # 7. Get Best of Iteration & Update Global Best Immediately
        bestIdx_iter, bestFitness_iter = -1, -float('inf')
        bestSPS_iter, bestInteract_iter = 0.0, 0.0
        try:
            # obtieneBest ahora usa el fitness basado en los scores recalculados
            bestIdx_iter, bestFitness_iter = operadorBacterial.obtieneBest(globalNFE)
            if 0 <= bestIdx_iter < numeroDeBacterias:
                # Leer el score SPS recalculado de self.blosumScore
                bestSPS_iter = operadorBacterial.blosumScore[bestIdx_iter] if isinstance(operadorBacterial.blosumScore[bestIdx_iter], (int, float)) else 0.0
                bestInteract_iter = operadorBacterial.tablaInteraction[bestIdx_iter] if isinstance(operadorBacterial.tablaInteraction[bestIdx_iter], (int, float)) else 0.0
                # Actualizar veryBest si es necesario (usando deepcopy corregido)
                if bestFitness_iter is not None and numpy.isfinite(bestFitness_iter) and \
                   (veryBest[1] is None or bestFitness_iter > veryBest[1]):
                     print(f"*** Nuevo Mejor Global (Iter {it + 1}): Idx {bestIdx_iter}, Fit {bestFitness_iter:.4f} (SPS: {bestSPS_iter:.1f}) ***")
                     veryBest[0] = bestIdx_iter; veryBest[1] = bestFitness_iter
                     try:
                         managed_list_item = poblacion[bestIdx_iter]
                         best_alignment_list = [list(seq) for seq in managed_list_item]
                         veryBest[2] = copy.deepcopy(best_alignment_list)
                         # --- DEBUG Check ---
                         # sps_check = operadorBacterial.calcular_sps_afin(veryBest[2])
                         # print(f"DEBUG: Copied alignment SPS check (Iter {it+1}, Idx {bestIdx_iter}): {sps_check:.1f} vs Reported: {bestSPS_iter:.1f}")
                         # ---
                     except Exception as e: print(f"Error deepcopy: {e}"); veryBest[2] = None
            else: bestIdx_iter = -1
        except Exception as e: print(f"Error en obtieneBest o actualización veryBest: {e}"); bestIdx_iter = -1

        # --- Reporting Iteration Results ---
        iter_end_time = time.time(); iter_time_seconds = iter_end_time - iter_start_time
        if csv_writer:
             try:
                 row_data = [ it + 1, bestIdx_iter if bestIdx_iter != -1 else "N/A", f"{bestFitness_iter:.4f}" if bestIdx_iter != -1 else "N/A",
                     f"{bestSPS_iter:.1f}" if bestIdx_iter != -1 else "N/A", f"{bestInteract_iter:.4f}" if bestIdx_iter != -1 else "N/A",
                     globalNFE, f"{iter_time_seconds:.4f}" ]
                 csv_writer.writerow(row_data)
             except Exception as write_e: print(f"Error escribiendo reporte iter {it+1}: {write_e}")

        # 8. Reproduction (Replace Worst with Global Best)
        try:
             if veryBest[0] is not None: operadorBacterial.replaceWorst(poblacion, veryBest[0])
        except Exception as e: print(f"Error replaceWorst: {e}")

        # --- 9. Elimination-Dispersion Step ---
        if (it + 1) % N_ed == 0:
             print(f"[{it+1}] === Ejecutando Eliminación-Dispersión ===")
             # ... (Lógica de E-D sin cambios) ...
             try:
                 fitness_list = list(operadorBacterial.tablaFitness)
                 valid_fitness = [(fit, idx) for idx, fit in enumerate(fitness_list) if isinstance(fit, (int, float)) and numpy.isfinite(fit)]
                 if len(valid_fitness) > num_dispersar:
                      valid_fitness.sort(key=lambda item: item[0])
                      indices_to_disperse = [idx for fit, idx in valid_fitness[:num_dispersar]]
                      print(f"Dispersando bacterias con índices: {indices_to_disperse}")
                      best_alignment_copy_for_dispersion = None
                      if veryBest[2] is not None: best_alignment_copy_for_dispersion = copy.deepcopy(veryBest[2])
                      elif bestIdx_iter != -1:
                           try:
                                managed_list_fallback = poblacion[bestIdx_iter]
                                list_fallback = [list(s) for s in managed_list_fallback]
                                best_alignment_copy_for_dispersion = copy.deepcopy(list_fallback)
                           except Exception: pass
                      for idx_disp in indices_to_disperse:
                           if best_alignment_copy_for_dispersion:
                                dispersed_alignment = copy.deepcopy(best_alignment_copy_for_dispersion)
                                for _ in range(nado_steps * 2):
                                     op_disp = random.choice([operadorBacterial._insert_gap, operadorBacterial._delete_gap, operadorBacterial._shift_gap, operadorBacterial._mover_bloque_gaps, operadorBacterial._mover_bloque_residuos])
                                     seq_idx_disp = random.randint(0, numSec - 1)
                                     try:
                                           if op_disp == operadorBacterial._mover_bloque_residuos: op_disp(dispersed_alignment, seq_idx_disp, block_size=random.randint(5,10))
                                           else: op_disp(dispersed_alignment, seq_idx_disp)
                                     except Exception: pass
                                operadorBacterial.cuadra_individual(dispersed_alignment)
                                poblacion[idx_disp] = dispersed_alignment
                           else:
                                print(f"Advertencia: Re-inicializando bacteria {idx_disp} en E-D.")
                                max_l = max((len(s) for s in secuencias_listas), default=0)
                                init_a = [];
                                for s in secuencias_listas: a_s = list(s); gap_c = max_l - len(a_s);
                                if gap_c > 0: a_s.extend(['-']*gap_c); init_a.append(a_s)
                                poblacion[idx_disp] = init_a
             except Exception as ed_e: print(f"Error durante Eliminación-Dispersión: {ed_e}")

        # 10. Reset Intermediate Lists (AFTER E-D)
        try: operadorBacterial.resetListas(numeroDeBacterias)
        except Exception as e: print(f"Error resetListas: {e}")

    # --- End of Loop ---
    overall_end_time = time.time()
    print("\n--- Ejecución BFOA (SPS Afín, Nuevos Ops, ED, Recalc) Finalizada ---")
    if report_file:
        try: report_file.close(); print(f"Reporte '{report_filename}' cerrado.")
        except Exception: pass

    # --- Final Results ---
    print("\n=== Mejor Solución Global Encontrada ===")
    if veryBest[1] is not None:
        print(f"Índice (última vez mejor): {veryBest[0]}")
        print(f"Fitness: {veryBest[1]:.4f}")
        if veryBest[2] and isinstance(veryBest[2], list):
             final_best_alignment = veryBest[2]
             print("Recalculando puntaje SPS Afín final...")
             try:
                 # --- LLAMADA FINAL A RECALCULAR ---
                 final_sps_score = operadorBacterial.calcular_sps_afin(final_best_alignment)
                 print(f"Puntaje SPS Afín (recalculado veryBest): {final_sps_score:.1f}")
                 # --- FIN LLAMADA FINAL ---
                 print("\nMejor Alineamiento:")
                 current_names = list(names)
                 if len(final_best_alignment) == len(current_names):
                     for i, seq_list in enumerate(final_best_alignment):
                          if isinstance(seq_list, list): print(f">{current_names[i]}\n{''.join(seq_list)}")
                          else: print(f">{current_names[i]}\nERROR: Data invalid.")
                 else: print("ERROR: Discrepancia secuencia/nombre o formato inválido.")
             except Exception as e: print(f"Error procesando alineamiento final: {e}")
        else: print("No se almacenó alineamiento válido en veryBest[2].")
    else: print("No se encontró solución válida.")
    print(f"\nNFE Total (evaluaciones interacción): {globalNFE}")
    print(f"Tiempo Total Ejecución: {overall_end_time - start_time:.2f} segundos")