
# test-weights test for two weighted distributions

def test_weights(data1, data2, wei1, wei2, npix_on, npix_off):

         # Create the energy histogram

         ebins = np.logspace(2.5,5.0,20)
         norm1 = np.sum(wei1)
         norm2 = np.sum(wei2)
         wei1_norm = weig1/norm1
         wei2_norm = weig2/norm2

         Wi_on, edges = np.histogram(data1,bins=ebins,weights=wei1_norm)
         S2i_on, edges = np.histogram(data1,bins=ebins,weights=np.power(wei1_norm,2))

         Wi_off, edges = np.histogram(data2,bins=ebins,weights=wei2_norm)
         S2i_off, edges = np.histogram(data2,bins=ebins,weights=np.power(wei2_norm,2))

         # Check for zero or very small denominators
         di2 = Wi_off*(Si2_on/Wi_on + Si2_off/Wi_off) # Eq. 10
         di2 = np.where(di2 == 0, np.inf, di2)

         # Calculate chi-squared sum with handled denominators
         chi2sum = 0
         chi2sum = np.sum(np.power((Wi_on - Wi_off), 2) / di2

         chi2sum_red = chi2sum/(len(ebins)-1)



         if 0.0 <= chi2sum_red <= 10.0:

         plt.errorbar(edges[1:],ehist_on/np.sum(ehist_on),yerr=sigma_on/np.sum(ehist_on),label='$N_{on}$')

         plt.errorbar(edges[1:],ehist_off/np.sum(ehist_off),yerr=sigma_off/np.sum(ehist_off),label='$N_{off}$')

         plt.loglog()

         plt.legend()

         plt.xlabel("Energy bins")

         plt.ylabel("ehist/np.sum(ehist)")

         fig = pylab.figure(1)

         fig.savefig('/home/aamarinp/Documents/ptracing-CosmicRay-analysis/figs/test_weights_phy_ind_minus1_final_pix_20bins/test_weights_phy_ind_minus1_final_pix_20bins_'+str(round(chi2sum_red,1))+'.png', dpi=250)

         plt.close()



         print('chi2sum', chi2sum)

         print('chi2sum_red', chi2sum_red)

         return chi2sum_red
