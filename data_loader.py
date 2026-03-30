import numpy as np

def load_data():
    allacts = []
    allfreqs = []
    allfields = []
    alleigs = []
    allrands = []


    allacts.append(np.load("activities.npy"))
    allfreqs.append(np.load("qfreqs.npy"))
    allfields.append(np.load('fields.npy'))
    alleigs.append(np.load('qeigs.npy'))
    allrands.append(np.load("used.npy"))


    return allfreqs, allacts, allfields, alleigs, allrands



def sort_spectra(allfreqs, allacts):
    

    sig = 29.72399

    activities = np.concatenate(allacts)
    freqs = np.concatenate(allfreqs)


    xx = np.round(np.linspace(900, 4000, 621))
    raw = [[],[],[],[]]
    iso = [[],[],[],[]]
    for x in xx:
        for j in range(4):
            wh = np.where(np.abs(freqs[:,j] - x) < 2.5)
            vv = np.sum(activities[wh][:,j])
            #iso[j].append(vv)
            dis = np.abs(freqs[:,j] - x)
            inten = np.sum(np.exp(-(dis/sig)**2)*activities[:,j])
            iso[j].append(inten)
            inten = np.sum(np.exp(-(dis/sig)**2))
            raw[j].append(inten)


    iso = np.asarray(iso)
    raw = np.asarray(raw)
    aniso = [[],[],[],[]]
    
    for x in xx:
        for j in range(4):
            wh = np.where(np.abs(freqs[:,j] - x) < 2.5)
            vv = np.sum(activities[wh][:,j+4])

            dis = np.abs(freqs[:,j] - x)
            inten = np.sum(np.exp(-(dis/sig)**2)*activities[:,j+4])
            aniso[j].append(inten)


    aniso = np.asarray(aniso)
    combo = np.sum(aniso*21/90 + iso/9, axis=0)


    ir = [[],[],[],[]]
    
    for x in xx:
        for j in range(4):
            wh = np.where(np.abs(freqs[:,j] - x) < 2.5)
            vv = np.sum(activities[wh][:,j+4])

            dis = np.abs(freqs[:,j] - x)
            inten = np.sum(np.exp(-(dis/sig)**2)*activities[:,j+8])
            ir[j].append(inten)


    ir1 = np.asarray(ir)
    oneir = np.sum(ir1, axis=0)

    ir = [[],[],[],[]]
    
    for x in xx:
        for j in range(4):
            wh = np.where(np.abs(freqs[:,j] - x) < 2.5)
            vv = np.sum(activities[wh][:,j+4])

            dis = np.abs(freqs[:,j] - x)
            inten = np.sum(np.exp(-(dis/sig)**2)*activities[:,j+12])
            ir[j].append(inten)


    ir2 = np.asarray(ir)
    twoir = np.sum(ir2, axis=0)




    combos = [xx, np.sum(iso, axis=0), np.sum(aniso, axis=0), combo, oneir, twoir, np.sum(raw, axis=0)]
    unsorted = [xx, iso, aniso, ir1, ir2]
    return combos, unsorted
