import numpy as np
import jax
import jax.numpy as jnp

import time

@jax.jit
def probability_for_randomized_matrix_multiply(matA, matB):
    matA_column_2norm = jax.vmap(lambda x: jnp.linalg.norm(x, ord = 2), in_axes=1)(matA)
    matB_row_2norm = jax.vmap(lambda x: jnp.linalg.norm(x, ord = 2), in_axes=0)(matB)
    temp = jax.vmap(lambda x, y: x * y)(matA_column_2norm, matB_row_2norm)
    return temp / jnp.sum(temp)

def randomized_matrix_multiply(matA, matB, n, c, prob, rand_key):
    pick = jax.random.choice(rand_key, jnp.arange(n), (c,), replace=True, p=prob)
    matC = jax.vmap(lambda x, y: x / jnp.sqrt(c * y), in_axes = (1, 0), out_axes=1)(matA[:, pick], prob[pick])
    matR = jax.vmap(lambda x, y: x / jnp.sqrt(c * y), in_axes = (0, 0))(matB[pick, :], prob[pick])
    return np.matmul(matC, matR)

def main():
    matA_row_num = 9999
    matA_column_num = 9998
    matB_row_num = matA_column_num
    matB_column_num = 9997

    c = 5000

    key_A = jax.random.PRNGKey(0)
    key_B = jax.random.PRNGKey(1)

    matA = jax.random.normal(key_A, shape =(matA_row_num, matA_column_num))
    matB = jax.random.normal(key_B, shape =(matB_row_num, matB_column_num))

    AB_start_time = time.perf_counter()

    matAB = np.matmul(matA, matB)

    AB_end_time = time.perf_counter()

    prob = probability_for_randomized_matrix_multiply(matA, matB)

    CR_start_time = time.perf_counter()

    matCR = randomized_matrix_multiply(matA, matB, matA_column_num, c, prob, jax.random.PRNGKey(2))

    CR_end_time = time.perf_counter()

    print("normal matrix multiplication time: " + str(AB_end_time - AB_start_time))
    print("prob time: " + str(CR_start_time - AB_end_time))
    print("randomized matrix multiplication " + "c = " + str(c) + " time: " + str(CR_end_time - CR_start_time))

    matAB_norm = jnp.linalg.norm(matAB, "fro")

    matCR_norms = []
    matE_norms = []
    for i in range(10):
        matCR = randomized_matrix_multiply(matA, matB, matA_column_num, c, prob, jax.random.PRNGKey(i))
        matCR_norms.append(jnp.linalg.norm(matCR, "fro"))
        matE_norms.append(jnp.linalg.norm(matAB-matCR, "fro"))

    print("normal matrix multiplication norm: " + str(matAB_norm))
    print("randomized matrix multiplication mean(norm): " + str(jnp.mean(jnp.asarray(matCR_norms))))
    print("randomized matrix multiplication var(norm): " + str(jnp.var(jnp.asarray(matCR_norms))))
    print("randomized matrix multiplication sigma: " + str(jnp.sqrt(jnp.var(jnp.asarray(matCR_norms)))))
    print("error matrix mean(norm): " + str(jnp.mean(jnp.asarray(matE_norms))))
    print("error matrix var(norm): " + str(jnp.var(jnp.asarray(matE_norms))))
    print("error matrix sigma: " + str(jnp.sqrt(jnp.var(jnp.asarray(matE_norms)))))

#    for c in range(0, matA_column_num, 100):
#
#        CR_start_time = time.perf_counter()
#
#        matCR = randomized_matrix_multiply(matA, matB, matA_column_num, c, prob, jax.random.PRNGKey(2))
#
#        CR_end_time = time.perf_counter()
#
#        print("randomized matrix multiplication " + "c = " + str(c) + " time: " + str(CR_end_time - CR_start_time))

    return 0

if __name__ == "__main__":
    main()