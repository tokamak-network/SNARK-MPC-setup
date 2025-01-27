<!--
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
-->

# Data format of MPC files
This document defines the format for the representation of the binary files produced in the MPC ceremony for Groth16 zk-SNARK parameter generation.  

## A. Challenge file

The challenge file structure is shown below in bytes for **compressed** form.

````
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 64                ┃  hash (BLAKE2b)                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ (2ⁿ⁺¹ - 1) * L    ┃  tau_powers_g1                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ (2ⁿ) * 2L         ┃  tau_powers_g2                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ (2ⁿ) * L          ┃  alpha_tau_powers_g1             ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ (2ⁿ) * L          ┃  beta_tau_powers_g1              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 2L                ┃  beta_g2                         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
````
where $n$ represents the required power for some circuit sizes, and the generated parameter are defined as following:

```math
tau_powers_g1 $= \tau^i \cdot G_1 \mid i \in \{0, 1, 2, \dots, 2^{n+1} - 2\}$

tau_powers_g2 $= \tau^i \cdot G_2 \mid i \in \{0, 1, 2, \dots, 2^n - 1\}$

alpha_tau_powers_g1 $= \alpha \cdot \tau^i \cdot G_1 \mid i \in \{0, 1, 2, \dots, 2^n - 1\}$

beta_tau_powers_g1 $= \beta \cdot \tau^i \cdot G_1 \mid i \in \{0, 1, 2, \dots, 2^n - 1\}$

beta\_g2 $= \beta \cdot G_2$
```

The point sizes of $G1$ and $G2$ are defined as follows:

|  | **Compressed** | **Uncompressed** |
| --- | --- | --- |
| **`G1_POINT_SIZE`** | $L$ | $2L$ |
| **`G2_POINT_SIZE`** | $2L$ | $4L$ |

The challenge file structure is shown below in bytes for **uncompressed** form.

````
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 64                ┃ hash (BLAKE2b)                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ (2ⁿ⁺¹ - 1) * 2L   ┃  tau_powers_g1                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ (2ⁿ) * 4L         ┃  tau_powers_g2                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ (2ⁿ) * 2L         ┃  alpha_tau_powers_g1             ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ (2ⁿ) * 2L         ┃  beta_tau_powers_g1              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 4L                ┃  beta_g2                         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
````

### Table representation

| No | Data  | Size (bytes) (**Compressed**) | Size (bytes) (**Uncompressed**) | **Description** |
| --- | --- | --- | --- | --- |
| 1. | hash (BLAKE2b) | 64 | 64 | `HASH_SIZE` |
| 2. | tau_powers_g1 | $(2^{n+1}-1)*L$ | $(2^{n+1}-1)*2L$ | `TAU_POWERS_G1_LENGTH × G1_POINT_SIZE` |
| 3. | tau_powers_g2 | $(2^{n})*2L$ | $(2^{n})*4L$ | `TAU_POWERS_LENGTH × G2_POINT_SIZE` |
| 4. | alpha_tau_powers_g1 | $(2^{n})*L$ | $(2^{n})*2L$ | `TAU_POWERS_LENGTH × G1_POINT_SIZE` |
| 5. | beta_tau_powers_g1  | $(2^{n})*L$ | $(2^{n})*2L$ | `TAU_POWERS_LENGTH × G1_POINT_SIZE` |
| 6. | beta_g2  | $2L$ | $4L$ | `G2_POINT_SIZE` |

### For example:

Below, we provide an example calculation for the parameters and total challenge file size as follows:

1. **`REQUIRED_POWER`** $= n$

    $n = 11$
2. **`TAU_POWERS_LENGTH` $= 2^n$**
    
    TAU_POWERS_LENGTH = $2^{n} = 2^{11}=2048$
    
    This is the number of powers of τ used in G2, and for alpha and beta in G1.
    
2. **`TAU_POWERS_G1_LENGTH`$= 2^{n+1}-1$**
    
    TAU_POWERS_G1_LENGTH = $2^{n+1}-1 = 2^{12}-1 = 4095$
    
    This is the number of powers of τ used in **G1**.
    
3. **`G1_POINT_SIZE`**
    - **Compressed:** $L$ = `32 bytes`
    - **Uncompressed:** $2L$ = `64 bytes`
    
    This size determines how many bytes are used to store a point in the G1 group.
    
4. **`G2_POINT_SIZE`**
    - **Compressed:** $2L$ = `64 bytes`
    - **Uncompressed:** $4L$ = `128 bytes`
    
    This size determines how many bytes are used to store a point in the G2 group.

With the example settings provided above, the table below presents the size of each data type and the total packet size:

| Data| Size (Bytes) |
| --- | --- |
| hash (BLAKE2b) | 64 |
| tau_powers_g1 | 262,080 |
| tau_powers_g2 | 262,144 |
| alpha_tau_powers_g1 | 131,072 |
| beta_tau_powers_g1 | 131,072 |
| beta_g2 | 128 |
| **total** | **758,560** |

## B. Response file

The response file structure is shown below in bytes for **compressed** form.
````
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 64                   ┃ hash (BLAKE2b of the challenge file) ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ (2ⁿ⁺¹ − 1) * L       ┃ tau_powers_g1                        ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ (2ⁿ) * 2L            ┃ tau_powers_g2                        ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ (2ⁿ) * L             ┃ alpha_tau_powers_g1                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ (2ⁿ) * L             ┃ beta_tau_powers_g1                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 2L                   ┃ beta_g2                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 18L       ┃ Public Key (Proof of contributor's secret key)  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
````

### Public Key Components

| **Component** | **Description** |
| --- | --- |
| **G1 Elements** | 6 points:  $(g^s, g^{s\tau}), (g^s, g^{s\alpha}), (g^s, g^{s\beta}).$ |
| **G2 Elements** | 3 points:  $H(g^s)  \text{ for }  \tau, \alpha, \beta.$ |
| **Purpose** | Proves knowledge of $\tau, \alpha, \beta.$ |
| **Verification** | Used in pairing-based checks during `verify_transform.rs`. |

The file structure is shown below in bytes for **uncompress** form.

````
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 64                   ┃ hash (BLAKE2b of the challenge file) ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ (2ⁿ⁺¹ − 1) * 2L      ┃ tau_powers_g1                        ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ (2ⁿ) * 4L            ┃ tau_powers_g2                        ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ (2ⁿ) * 2L            ┃ alpha_tau_powers_g1                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ (2ⁿ) * 2L            ┃ beta_tau_powers_g1                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 4L                   ┃ beta_g2                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 36L       ┃ Public Key (Proof of contributor's secret key)  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
````

## C. Transcript file

The file is the combination of response_old and response files. Its size is double of a response file. The below command is used to combine `response_old` and `response` into a single `transcript` file and it effectively merges the content.

`Get-Content response_old, response | Add-Content transcript`

## D. `phase1radix2mX` file

The file structure is shown below in bytes for **compressed** form.
````
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ (2ⁿ⁻¹) * 2L        ┃ G1 Lagrange Coefficients        ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ (2ⁿ⁻¹) * 2L        ┃ G2 Lagrange Coefficients        ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ (2ⁿ⁻¹) * L         ┃ alpha Powers                    ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ (2ⁿ⁻¹)* L          ┃ beta Powers                     ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ (2ⁿ⁻¹ − 1) * L     ┃ H Query                         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
````

## E. `mimc.params` file

The `mimc.params` file includes the following parameters:

### Proving Key

- `a`: Vector of G1 elements for the QAP A query.
- `b_g1`: Vector of G1 elements for the QAP B query (G1 part).
- `b_g2`: Vector of G2 elements for the QAP B query (G2 part).
- `h`: Vector of G1 elements representing the coefficients of the quotient polynomial H(x).
- `l`: Vector of G1 elements for the Lagrange interpolation of the circuit's constraints.

### Verification Key

- `alpha_g1`: G1 group element for the Groth16 alpha parameter.
- `beta_g1`: G1 group element for the Groth16 beta parameter.
- `beta_g2`: G2 group element for the Groth16 beta parameter.
- `gamma_g2`: G2 group element for the Groth16 gamma parameter.
- `delta_g1`: G1 group element for the Groth16 delta parameter.
- `delta_g2`: G2 group element for the Groth16 delta parameter.
- `IC`: Vector of G1 elements for public input coefficients.

The file structure for `mimc.params` file is shown below.
````
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Verifying Key (VK)                                    ┃
┃   L                  ┃  Alpha (G1)                    ┃
┃   3L                 ┃  Beta  (G1, G2)                ┃
┃   2L                 ┃  Gamma (G2)                    ┃
┃   3L                 ┃  Delta (G1, G2)                ┃
┃   num_inputs         ┃  Input Coefficients (IC) in G1 ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Proving Key (PK)                                      ┃
┃   h_query_size       ┃  H Query (G1)                  ┃
┃   l_query_size       ┃  L Query (G1)                  ┃
┃   a_query_size       ┃  A Query (G1)                  ┃
┃   b_g1_query_size    ┃  B Query in G1                 ┃
┃   b_g2_query_size    ┃  B Query in G2                 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
````
