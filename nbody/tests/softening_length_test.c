//This test is designed to test the softening length array calculations for various different component profiles
//If a new dwarf profile is added, please add a corresponding test case below
//This test will also run a two component Nbody to make sure the array is correctly passed 
//(single component is tested in model tests)
//Like with the model tests, the expected result may need to be updated if changes are made to the simulation
#include "test_env_util.h"
#include "nbody_lua_models.h"
#include "nbody_potential_types.h"

int check_result(real* eps_array, real eps[3]) {
    for (int i = 0; i < 3; i++) {
        if (eps_array[i] != eps[i]) {
            return 0;
        }
    }
    return 1;
}

int main() {

    //Check softening lengths for various models. These are the same params as used in mixeddwarf test

    real eps_l;
    real eps_cross;
    real eps_d;
    real eps[3];
    real* eps_array;
    unsigned int lm_nbody = 10000;
    unsigned int nbody = 40000;
    Dwarf* light_comp;
    Dwarf* dark_comp;
    int failed = 0;

    light_comp->mass = 12;
    light_comp->scaleLength = 0.2;
    dark_comp->mass = 48;
    dark_comp->scaleLength = 0.8;

    //Plummer-Plummer
    eps_l = 0.00000810512565372;
    eps_cross = 0.00000810512565372;
    eps_d = 0.00000239956397623;

    eps[0] = eps_l;
    eps[1] = eps_cross;
    eps[2] = eps_d;

    light_comp->type = 0;
    dark_comp->type = 0;

    eps_array = nbCalculateEps2_NEW(light_comp, dark_comp, lm_nbody, nbody); 

    if (!check_result(eps_array, eps)) {
        mw_printf("Test failed for Plummer-Plummer model\n");
        mw_printf("Expected: %.80f, %.80f, %.80f\n", eps[0], eps[1], eps[2]);
        mw_printf("Got: %.80f, %.80f, %.80f\n", eps_array[0], eps_array[1], eps_array[2]);
        failed = 1;
    }

    //Plummer-Hernquist
    eps_l = 0.0000108469691349;
    eps_cross = 0.0000108469691349;
    eps_d = 0.00000640401178546;

    eps[0] = eps_l;
    eps[1] = eps_cross;
    eps[2] = eps_d;

    dark_comp->type = 2;

    eps_array = nbCalculateEps2_NEW(light_comp, dark_comp, lm_nbody, nbody); 

    if (!check_result(eps_array, eps)) {
        mw_printf("Test failed for Plummer-Hernquist model\n");
        mw_printf("Expected: %.80f, %.80f, %.80f\n", eps[0], eps[1], eps[2]);
        mw_printf("Got: %.80f, %.80f, %.80f\n", eps_array[0], eps_array[1], eps_array[2]);
        failed = 1;
    }

    //Plummer-NFW
    eps_l = 0.0;
    eps_cross = 0.0;
    eps_d = 0.0;

    eps[0] = eps_l;
    eps[1] = eps_cross;
    eps[2] = eps_d;

    dark_comp->type = 1;

    eps_array = nbCalculateEps2_NEW(light_comp, dark_comp, lm_nbody, nbody); 

    if (!check_result(eps_array, eps)) {
        mw_printf("Test failed for Plummer-NFW model\n");
        mw_printf("Expected: %.80f, %.80f, %.80f\n", eps[0], eps[1], eps[2]);
        mw_printf("Got: %.80f, %.80f, %.80f\n", eps_array[0], eps_array[1], eps_array[2]);
        failed = 1;
    }

    //Plummer-Cored
    eps_l = 0.0;
    eps_cross = 0.0;
    eps_d = 0.0;

    eps[0] = eps_l;
    eps[1] = eps_cross;
    eps[2] = eps_d;

    dark_comp->type = 4;
    dark_comp->r1 = 0.7;
    dark_comp->rc = 0.6;

    eps_array = nbCalculateEps2_NEW(light_comp, dark_comp, lm_nbody, nbody); 

    if (!check_result(eps_array, eps)) {
        mw_printf("Test failed for Plummer-cored model\n");
        mw_printf("Expected: %.80f, %.80f, %.80f\n", eps[0], eps[1], eps[2]);
        mw_printf("Got: %.80f, %.80f, %.80f\n", eps_array[0], eps_array[1], eps_array[2]);
        failed = 1;
    }

    //Plummer-Cutoff-Cored
    eps_l = 0.0;
    eps_cross = 0.0;
    eps_d = 0.0;

    eps[0] = eps_l;
    eps[1] = eps_cross;
    eps[2] = eps_d;

    dark_comp->rcut = 4.5;

    eps_array = nbCalculateEps2_NEW(light_comp, dark_comp, lm_nbody, nbody); 

    if (!check_result(eps_array, eps)) {
        mw_printf("Test failed for Plummer-cutoff-cored model\n");
        mw_printf("Expected: %.80f, %.80f, %.80f\n", eps[0], eps[1], eps[2]);
        mw_printf("Got: %.80f, %.80f, %.80f\n", eps_array[0], eps_array[1], eps_array[2]);
        failed = 1;
    }

    //Plummer-Cutoff-NFW
    eps_l = 0.0;
    eps_cross = 0.0;
    eps_d = 0.0;

    eps[0] = eps_l;
    eps[1] = eps_cross;
    eps[2] = eps_d;

    dark_comp->type = 1;
    dark_comp->r1 = 0.0;
    dark_comp->rc = 0.0;

    eps_array = nbCalculateEps2_NEW(light_comp, dark_comp, lm_nbody, nbody); 

    if (!check_result(eps_array, eps)) {
        mw_printf("Test failed for Plummer-cutoff-NFW model\n");
        mw_printf("Expected: %.80f, %.80f, %.80f\n", eps[0], eps[1], eps[2]);
        mw_printf("Got: %.80f, %.80f, %.80f\n", eps_array[0], eps_array[1], eps_array[2]);
        failed = 1;
    }

    return failed;
}

