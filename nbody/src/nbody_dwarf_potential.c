/*
 * Copyright (c) 2012 Rensselaer Polytechnic Institute
 * Copyright (c) 2016-2018 Siddhartha Shelton
 * 
 * This file is part of Milkway@Home.
 *
 * Milkyway@Home is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Milkyway@Home is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Milkyway@Home.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "nbody_dwarf_potential.h"
#include "milkyway_math.h"
#include "nbody_types.h"
#include "nbody_potential_types.h"
#include "nbody_mass.h"

/* NOTE
 * we want the term nu which is the density per mass unit. However, these return just normal density.
 * In galactic dynamics 2nd edition, equation 4.48 defines nu which the distribution function is written 
 * in terms of. However, this mass is not the mass of each component but the total mass of both. Therefore,
 * this term can be pulled out of the integral. since we are rejection sampling, it cancels with the denom. 
 * It does not change the distribution so it would be ok if it had not canceled.
 * the potential functions return the negative version of the potential, psi, which is what is needed. 
 */


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*                             PLUMMER                                                                                   */
/* this potential and density are both taken from binney 2nd ed                                                          */
 static real plummer_den(const Dwarf* model, real r)                                                                     //
{                                                                                                                        //
    const real mass = model->mass;                                                                                       //
    const real rscale = model->scaleLength;                                                                              //
    return  (3.0 / (4.0 * M_PI)) * (mass / cube(rscale)) * minusfivehalves( (1.0 + sqr(r / rscale)) ) ;                  //
}                                                                                                                        //
                                                                                                                         //
 static real plummer_pot(const Dwarf* model, real r)                                                                     //
{                                                                                                                        //
    const real mass = model->mass;                                                                                       //
    const real rscale = model->scaleLength;                                                                              //
    return mass / mw_sqrt(sqr(r) + sqr(rscale));                                                                         //
}                                                                                                                        //
                                                                                                                         //
 static real plummer_vel_disp(const Dwarf* model, real r)                                                                //
{                                                                                                                        //
    const real mass = model->mass;                                                                                       //
    const real rscale = model->scaleLength;                                                                              //
    return mass / (6* mw_sqrt(sqr(r)+sqr(rscale)));                                                                      //
}                                                                                                                        //
 static real plummer_half_mass(const Dwarf* model)                                                                       //
 {                                                                                                                       //
    const real rscale = model->scaleLength;                                                                              //
    return 1 / mw_sqrt(1 / mw_pow(5, 2 / 3) - 1) * rscale;                                                               //
 }                                                                                                                       //
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*                            NFW                                                                                        */
/* this density is taken from the 1997 paper by nfw. the potential is taken from binney 2nd ed                           */
 static real nfw_den(const Dwarf* model, real r)                                                                         //
{                                                                                                                        //
    const real rscale = model->scaleLength;                                                                              //
    const real p0 = model->p0;                                                                                           //
    real R = r / rscale;                                                                                                 //
    /* at r = 0 the density goes to inf. however, the sampling is guarded against r = 0 anyway.*/                        //
    return p0 * inv(R) * inv(sqr(1.0 + R));                                                                              //
}                                                                                                                        //
                                                                                                                         //
 static real nfw_pot(const Dwarf* model, real r)                                                                         //
{                                                                                                                        //
    const real rscale = model->scaleLength;                                                                              //
    const real p0 = model->p0;                                                                                           //
    real R = r / rscale;                                                                                                 //
    /* at r = 0 the pot goes to inf. however, the sampling is guarded against r = 0 anyway. */                           //
    return  4.0 * M_PI * sqr(rscale) * p0 * inv(R) * mw_log(1.0 + R);                                                    //
}                                                                                                                        //
                                                                                                                         //
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*                             GENERAL HERNQUIST                                                                         */
/* this potential and density are both taken from the 1990 paper by hernquist                                            */
static real gen_hern_den(const Dwarf* model, real r)                                                                     //
{                                                                                                                        //
    const real mass = model->mass;                                                                                       //
    const real rscale = model->scaleLength;                                                                              //
    return inv(2.0 * M_PI) * mass * rscale / ( r * cube(r + rscale));                                                    //
}                                                                                                                        //
                                                                                                                         //
static real gen_hern_pot(const Dwarf* model, real r)                                                                     //
{                                                                                                                        //
    const real mass = model->mass;                                                                                       //
    const real rscale = model->scaleLength;                                                                              //
    return mass / (r + rscale);                                                                                          //
}                                                                                                                        //
                                                                                                                         //
static real gen_hern_vel_disp(const Dwarf* model, real r)                                                                //
{                                                                                                                        //
    const real mass = model->mass;                                                                                       //
    const real rscale = model->scaleLength;                                                                              //
    const real term1 = 12 * r * mw_pow(r + rscale, 3.0) * mw_log((r + rscale) / r) / mw_pow(rscale, 4.0);                //
    const real term2 = 25 + 52 * r / rscale + 42 * sqr(r / rscale) + 12 * mw_pow(r / rscale, 3.0);                       //
    return mass / (12*rscale) * (term1 - r / (r + rscale) * term2);                                                      //
}                                                                                                                        //
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*                             EINASTO                                                                                   */
/* these are taken from the einasto paper. There are many problems with this, so it is currently unused.                 */
static real einasto_den(const Dwarf* model, real r)                                                                      //
{                                                                                                                        //
    const real mass __attribute__((unused)) = model->mass;                                                               //
    const real h = model->scaleLength;                                                                                   //
    const real n = model->n;                                                                                             //
                                                                                                                         //
    real coeff = 1.0 / ( 4.0 * M_PI * cube(h) * n * GammaFunc(3.0 * n));                                                 //
    real thing = mw_pow(r, inv(n));                                                                                      //
    return coeff * mw_exp(-thing);                                                                                       //
}                                                                                                                        //
                                                                                                                         //
static real einasto_pot(const Dwarf* model, real r)                                                                      //
{                                                                                                                        //
    const real mass = model->mass;                                                                                       //
    const real h = model->scaleLength;                                                                                   //
    const real n = model->n;                                                                                             //
                                                                                                                         //
    real coeff = mass / (h * r);                                                                                         //
    real thing = mw_pow(r, 1.0 / n);                                                                                     //
                                                                                                                         //
    real term1 = UpperIncompleteGammaFunc(3.0 * n, thing);                                                                    //
    real term2 = r * UpperIncompleteGammaFunc(2.0 * n, thing);                                                                //
    real term = 1.0 - ( term1 + term2 ) / GammaFunc(3.0 * n);                                                            //
    return coeff * term;                                                                                                 //
}                                                                                                                        //
                                                                                                                         //
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*                             CORED                                                                                     */
/* this potential and density are cored profiles to be used with SIDM.                                                   */
static real cored_den(const Dwarf* model, real r)                                                                        //
{                                                                                                                        //
    const real r1 = model->r1;                                                                                           //
    real p = 0.0;                                                                                                        //
    real rscale = 0.0;                                                                                                   //
    if(r <= r1)                                                                                                          //
    {                                                                                                                    //
        p = model->p0;                                                                                                   //
        rscale = model->rc;                                                                                              //
        return p / (1.0 + sqr(r / rscale));                                                                              //
    }                                                                                                                    //
    else                                                                                                                 //
    {                                                                                                                    //
        p = model->ps;                                                                                                   //
        rscale = model->scaleLength;                                                                                     //
        return p / ((r / rscale) * sqr(1.0 + r / rscale));                                                               //
    }                                                                                                                    //
}                                                                                                                        //
                                                                                                                         //
static real cored_pot(const Dwarf* model, real r)                                                                        //
{                                                                                                                        //
    const real r1 = model->r1;                                                                                           //
    const real p0 = model->p0;                                                                                           //
    const real rc = model->rc;                                                                                           //
    const real ps = model->ps;                                                                                           //
    const real rs = model->scaleLength;                                                                                  //
    const real C3 = 4.0 * M_PI * (                                                                                       //
            ps * cube(rs) * (                                                                                            //
                mw_log((1.0 + r1 / rs)) - r1 / (rs + r1)                                                                 //
            )                                                                                                            //
            - p0 * sqr(rc) * (                                                                                           //
                r1 - rc * mw_atan(r1 / rc)                                                                               //
            )                                                                                                            //
        );                                                                                                               //
                                                                                                                         //
    if(r <= r1)                                                                                                          //
    {                                                                                                                    //
            const real C2 = C3 / r1 - (4.0 * M_PI) / r1 * (                                                              //
            ps * cube(rs) * mw_log(1 + r1 / rs) +                                                                        //
            p0 * (                                                                                                       //
                (sqr(rc) * r1) / 2.0 * mw_log(sqr(r1) + sqr(rc)) +                                                       //
                cube(rc) * mw_atan(r1 / rc)                                                                              //
            )                                                                                                            //
        );                                                                                                               //
        return -1.0 * (4.0 * M_PI * p0 * (                                                                               //
            sqr(rc) / 2.0 * mw_log(sqr(r) + sqr(rc)) +                                                                   //
            cube(rc) / r * mw_atan(r / rc)                                                                               //
        ) + C2);                                                                                                         //
    }                                                                                                                    //
    else                                                                                                                 //
    {                                                                                                                    //
            return  -1.0 * (-4.0 * M_PI * ps * cube(rs) / r * mw_log(1.0 + r / rs) + C3 / r);                            //
    }                                                                                                                    //
}                                                                                                                        //
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

real get_potential(const Dwarf* model, real r)
{
    real pot_temp = 0.0;

    switch(model->type)
    {
        case Plummer:
            pot_temp = plummer_pot(model, r);
            break;
        case NFW:
            pot_temp = nfw_pot(model, r );
            break;
        case General_Hernquist:
            pot_temp = gen_hern_pot(model, r );
            break;
        case Einasto:
            printf("WARNING: Einsato dwarf currently has problems and should not be used \n");
            pot_temp = einasto_pot(model, r);
            break;
        case Cored:
            pot_temp = cored_pot(model, r);
            break;
        case InvalidDwarf:
        default:
            mw_fail("Invalid dwarf type, %d\n", model->type);
    }

    return pot_temp;
}



real get_density(const Dwarf* model, real r)
{
    real den_temp = 0.0;

    switch(model->type)
    {
        case Plummer:
            den_temp = plummer_den(model, r);
            break;
        case NFW:
            den_temp = nfw_den(model, r );
            break;
        case General_Hernquist:
            den_temp = gen_hern_den(model, r );
            break;
        case Einasto:
            printf("WARNING: Einsato dwarf currently has problems and should not be used \n");
            den_temp = einasto_den(model, r);
            break;
        case Cored:
            den_temp = cored_den(model, r);
            break;
        case InvalidDwarf:
        default:
            mw_fail("Invalid dwarf type, %d\n", model->type);

    }

    return den_temp;
}

real get_hmr(const Dwarf* model) //radii calculated here are for softening length calculation
{
    real hmr_temp = 0;

    switch(model->type)
    {
        case Plummer:
            hmr_temp = 1.3*model->scaleLength;
            break;
        case NFW:
        // This is (allegedly) the point of maximum density, but it should be sufficient for softening length calculations
            hmr_temp = model->scaleLength;
            break;
        case General_Hernquist:
            hmr_temp = (1 + mw_sqrt(2))*model->scaleLength;
            break;
        case Einasto:
            hmr_temp = model->scaleLength;
            break;
        case Cored:
        // This is also the point of maximum density, but it should be sufficient for softening length calculations
            hmr_temp = (model->scaleLength > model->r1) ? model->scaleLength : model->r1;
            break;
        case InvalidDwarf:
        default:
            mw_fail("Invalid dwarf type, %d\n", model->type);
    }

    return hmr_temp;
}
