from PIL import Image
import torch
from .model import ResNet , Bottleneck
from collections import namedtuple
import torchvision.transforms as transforms
import json
import os
from django.conf import settings

# import torchvision.transforms.functional as TF
ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])

resnet50_config = ResNetConfig(block = Bottleneck,
                               n_blocks = [3, 4, 6, 3],
                               channels = [64, 128, 256, 512])
PATH_DATASET = r"./media/dataset/"
device = torch.device('cpu')
model = ResNet(config=resnet50_config, output_dim=12)
model.load_state_dict(torch.load("./model_leaf_plant_disease_detection_model_v1.pt", map_location = device))


model = model.to(device)
classes = ['Chili healthy','Chili leaf curl', 'Chili leaf spot', 'Chili whitefy','Chili yellowish', 'Corn(maize) Gray leaf spot',
'Corn(maize) Common rust', 'Corn(maize) healthy','Corn(maize) Northern Leaf Blight','Potato Early blight','Potato healthy','Potato Late blight']


pretrained_means = [0.4581, 0.5027, 0.3906]
pretrained_stds = [0.1798, 0.1671, 0.1852]
test_transforms = transforms.Compose([
                                      transforms.Resize(size=255),
                                      transforms.CenterCrop(size=244),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean = pretrained_means, 
                                          std = pretrained_stds)
                                      ])


def single_prediction(image_path):
    path = '.' + image_path; 
    image = Image.open(path)

    # get normalized image
    img_normalized = test_transforms(image).float()
    img_normalized = img_normalized.unsqueeze_(0)
    with torch.no_grad():
      model.eval()
      output =model(img_normalized)
      print(output)
      index = output.data.cpu().numpy().argmax()
      print("Original : ", image_path[22:-4])

      # predict image similar
      smallest_dist = float('inf')
      label_similiar = None
      for img in os.listdir(PATH_DATASET + f'train/{classes[index]}'):
        image_ = Image.open(PATH_DATASET + f'train/{classes[index]}/{img}')
        img_train = test_transforms(image_)
        euclid_dis = torch.cdist(test_transforms(image), img_train)
        euclid_dis = torch.mean(euclid_dis).item()
        if euclid_dis < smallest_dist:
          smallest_dist = euclid_dis
          label_similiar = img
      str = "/media/dataset/" + f'train/{classes[index]}/{label_similiar}'
      
      return classes[index], str

    # output = output.detach().numpy()
    # index = np.argmax(output)
    # return index
    # pred_csv = data["disease_name"][index]
    # print(pred_csv)

def get_solution(name, disease):
    
    jsonStr = r"""
      [
    {
        "name": "Apple",
        "disease": "Apple Scab",
        "oveview": [
            "A serious disease of apples and ornamental crabapples, apple scab (Venturia inaequalis) attacks both leaves and fruit. The fungal disease forms pale yellow or olive-green spots on the upper surface of leaves. Dark, velvety spots may appear on the lower surface. Severely infected leaves become twisted and puckered and may drop early in the summer.",
            "Symptoms on fruit are similar to those found on leaves. Scabby spots are sunken and tan and may have velvety spores in the center. As these spots mature, they become larger and turn brown and corky. Infected fruit becomes distorted and may crack allowing entry of secondary organisms. Severely affected fruit may drop, especially when young.",
            "https://www.planetnatural.com/pest-problem-solver/plant-disease/apple-scab"
        ],
        "solution":[
            "Choose resistant varieties when possible.",
            "Rake under trees and destroy infected leaves to reduce the number of fungal spores available to start the disease cycle over again next spring.",
            "Water in the evening or early morning hours (avoid overhead irrigation) to give the leaves time to dry out before infection can occur.",
            "Spread a 3- to 6-inch layer of compost under trees, keeping it away from the trunk, to cover soil and prevent splash dispersal of the fungal spores.",
            "For best control, spray liquid copper soap early, two weeks before symptoms normally appear. Alternatively, begin applications when disease first appears, and repeat at 7 to 10 day intervals up to blossom drop.",
            "Bonide® Sulfur Plant Fungicide, a finely ground wettable powder, is used in pre-blossom applications and must go on before rainy or spore discharge periods. Apply from pre-pink through cover (2 Tbsp/ gallon of water), or use in cover sprays up to the day of harvest.",
            "Organocide® Plant Doctor is an earth-friendly systemic fungicide that works its way through the entire plant to combat a large number of diseases on ornamentals, turf, fruit and more. Apply as a soil drench or foliar spray (3-4 tsp/ gallon of water) to prevent and attack fungal problems.",
            "Containing sulfur and pyrethrins, Bonide® Orchard Spray is a safe, one-hit concentrate for insect attacks and fungal problems. For best results, apply as a protective spray (2.5 oz/ gallon) early in the season. If disease, insects or wet weather are present, mix 5 oz in one gallon of water. Thoroughly spray all parts of the plant, especially new shoots."
        ]
},
    {
        "name": "Apple",
        "disease": "Black Rot",
        "oveview": [
            "Black rot is occasionally a problem on Minnesota apple trees. This fungal disease causes leaf spot, fruit rot and cankers on branches. Trees are more likely to be infected if they are: Not fully hardy in Minnesota, Infected with fire blight or Stressed by environmental factors like drought.",
            "Large brown rotten areas can form anywhere on the fruit but are most common on the blossom end. Brown to black concentric rings can often be seen on larger infections. The flesh of the apple is brown but remains firm. Infected leaves develop \"frog-eye leaf spot\". These are circular spots with purplish or reddish edges and light tan interiors.",
            "https://extension.umn.edu/plant-diseases/black-rot-apple"
        ],
        "solution":[
            "Prune out dead or diseased branches.",
            "Pick all dried and shriveled fruits remaining on the trees.",
            "Remove infected plant material from the area.",
            "All infected plant parts should be burned, buried or sent to a municipal composting site.",
            "Be sure to remove the stumps of any apple trees you cut down. Dead stumps can be a source of spores.",
            "Provide trees with adequate water.",
            "Keep fire blight in check.",
            "Remove any limbs or trees killed by fire blight to discourage black rot."
        ]
        },
    {
        "name": "Apple",
        "disease": "Cedar Rust",
        "oveview": [
            "Cedar apple rust (Gymnosporangium juniperi-virginianae) is a fungal disease that requires juniper plants to complete its complicated two year life-cycle. Spores overwinter as a reddish-brown gall on young twigs of various juniper species. In early spring, during wet weather, these galls swell and bright orange masses of spores are blown by the wind where they infect susceptible apple and crab-apple trees. The spores that develop on these trees will only infect junipers the following year. From year to year, the disease must pass from junipers to apples to junipers again; it cannot spread between apple trees.",
            "On apple and crab-apple trees, look for pale yellow pinhead sized spots on the upper surface of the leaves shortly after bloom. These gradually enlarge to bright orange-yellow spots which make the disease easy to identify. Orange spots may develop on the fruit as well. Heavily infected leaves may drop prematurely.",
            "https://www.planetnatural.com/pest-problem-solver/plant-disease/cedar-apple-rust"
        ],
        "solution":[
            "Choose resistant cultivars when available.",
            "Rake up and dispose of fallen leaves and other debris from under trees.",
            "Remove galls from infected junipers. In some cases, juniper plants should be removed entirely.",
            "Apply preventative, disease-fighting fungicides labeled for use on apples weekly, starting with bud break, to protect trees from spores being released by the juniper host. This occurs only once per year, so additional applications after this springtime spread are not necessary.",
            "On juniper, rust can be controlled by spraying plants with a copper solution (0.5 to 2.0 oz/ gallon of water) at least four times between late August and late October.",
            "Safely treat most fungal and bacterial diseases with SERENADE Garden. This broad spectrum bio-fungicide uses a patented strain of Bacillus subtilis that is registered for organic use. Best of all, SERENADE is completely non-toxic to honey bees and beneficial insects.",
            "Containing sulfur and pyrethrins, Bonide® Orchard Spray is a safe, one-hit concentrate for insect attacks and fungal problems. For best results, apply as a protective spray (2.5 oz/ gallon) early in the season. If disease, insects or wet weather are present, mix 5 oz in one gallon of water. Thoroughly spray all parts of the plant, especially new shoots.",
            "Contact your local Agricultural Extension office for other possible solutions in your area."
        ]
    },
    {
        "name": "Apple",
        "disease": "Healthy",
        "oveview": [
            "Healthy apple"
        ],
        "solution":[
            "Your apple are healthy. You took good care of it.",
            "Just take care of it as you usually do."
        ]
    },
    {
        "name": "Blueberry",
        "disease": "Healthy",
        "oveview": [
            "Healthy Crops"
        ],
        "solution":[
            "Your crops are healthy. You took good care of it.",
            "Just take care of it as you usually do."
        ]
    },
    {
        "name": "Cherry",
        "disease": "Powdery Mildew",
        "oveview": [
            "Powdery mildew of sweet and sour cherry is caused by Podosphaera clandestina, an obligate biotrophic fungus. Mid- and late-season sweet cherry (Prunus avium) cultivars are commonly affected, rendering them unmarketable due to the covering of white fungal growth on the cherry surface. Season long disease control of both leaves and fruit is critical to minimize overall disease pressure in the orchard and consequently to protect developing fruit from accumulating spores on their surfaces.",
        "Initial symptoms, often occurring 7 to 10 days after the onset of the first irrigation, are light roughly-circular, powdery looking patches on young, susceptible leaves (newly unfolded, and light green expanding leaves). Older leaves develop an age-related (ontogenic) resistance to powdery mildew and are naturally more resistant to infection than younger leaves. Look for early leaf infections on root suckers, the interior of the canopy or the crotch of the tree where humidity is high.",
        "http://treefruit.wsu.edu/crop-protection/disease-management/cherry-powdery-mildew"
        ],
        "solution":[
            "Abound at 12 to 15.5 fl oz/A. Do not apply with silicone-based surfactants. May be applied the day of harvest. Sprayers used for Abound should not be used on apples. Fair efficacy. Group 11 fungicide. 4-hr reentry.",
            "Cabrio EG at 9.5 oz/A. May be used at harvest. Excellent efficacy. Group 11 fungicide. 12-hr reentry.",
            "Cevya at 5 fl oz/A. May be used day of harvest. Group 3 fungicide. 12-hr reentry.",
            "Eagle 20 EW at 2 to 3 fl oz/100 gal water for home orchards or landscape use. Can be applied up to the day of harvest. Group 3 fungicide. 24-hr reentry.",
            "Flint Extra at 2.5 to 3.8 oz/A. Do not apply within 1 day of harvest. 'Concord' grapes may be injured if accidentally sprayed. Group 11 fungicide. 12-hr reentry.",
            "Fontelis at 14 to 20 fl oz/A. Can be used day of harvest. Group 7 fungicide. 12-hr reentry.",
            "Gatten at 6 to 8 fl oz/A. Do not use within 3 days of harvest. Group U13 fungicide. 12-hr reentry.",
            "Inspire at 7 fl oz/A. May be used day of harvest. Group 3 fungicide. 12-hr reentry.",
            "Inspire Super at 16 to 20 fl oz/A. Tart Cherry only. Do not apply within 2 days of harvest. Group 3 + 9 fungicide. 12-hr reentry.",
            "Luna Experience at 6 to 10 fl oz/A. May be used day of harvest. Group 3 + 7 fungicide. 12-hr reentry."
        ]
    },
     {
        "name": "Cherry",
        "disease": "Healthy",
        "oveview": [
            "Healthy Cherry"
        ],
        "solution":[
            "Your Cherry are healthy. You took good care of it.",
            "Just take care of it as you usually do."
        ]
    },
    {
        "name": "Corn(Maize)",
        "disease": "Gray Leaf Spot",
        "oveview": [
            "Gray leaf spot (GLS) is a common fungal disease in the United States caused by the pathogen Cercospora zeae-maydis in corn. Disease development is favored by warm temperatures, 80°F or 27 °C; and high humidity, relative humidity of 90% or higher for 12 hours or more. Cercospora zeae-maydis overwinters in corn residue, allowing inoculum to build up from year to year in fields. Cropping systems with reduced- or no-till and/or continuous corn are at higher risk for gray leaf spot outbreaks.",
        "Gray leaf spot lesions begin as small necrotic pinpoints with chlorotic halos, these are more visible when leaves are backlit. Coloration of initial lesions can range from tan to brown before sporulation begins. Because early lesions are ambiguous, they are easily confused with other foliar diseases such as anthracnose leaf blight, eyespot, or common rust. As infection progresses, lesions begin to take on a more distinct shape. Lesion expansion is limited by parallel leaf veins, resulting in the blocky shaped “spots”. As sporulation commences, the lesions take on a more gray coloration.",
        "https://www.pioneer.com/us/agronomy/gray_leaf_spot_cropfocus.html"
        ],
        "solution":[
            "Fungicides, if sprayed early in season before initial damage, can be effective in reducing disease.",
            "Headline EC is to be applied at 400–600 millilitres per hectare (0.17–0.26 US qt/acre). For optimal disease control, begin applications prior to disease development (see §Disease cycle). This fungicide can only be applied a maximum of 2 applications/year. Ground and aerial application are both acceptable",
            "Quilt is to be applied at 0.75–1.0 litre per hectare (0.32–0.43 US qt/acre). Application of Quilt is to be made upon first appearance of disease, followed by a second application 14 days after, if environmental conditions are favorable for disease development (see §Disease cycle). Upon browning of corn sheaths, Quilt is not to be applied. This fungicide can only be applied a maximum 2 applications/yr. Ground and aerial application are both acceptable.",
            "Proline 480 SC is to be applied at 420 millilitres per hectare (0.18 US qt/acre). This fungicide can only be applied a maximum 1 time/year. Only ground application is acceptable. A 24-hour re-entry time is required (minimum amount of time that must pass between the time a fungicide is applied to an area or crop and the time that people can go into that area without protective clothing and equipment)",
            "Tilt 250 is to be applied at 500 millilitres per hectare (0.21 US qt/acre). Bumper 418 EC is to be applied at 300 millilitres per hectare (0.13 US qt/acre). Both fungicides are to be applied when rust pustules first appear. If disease is prevalent after primary application, a second application 14 days later may be necessary. Two weeks later, a third application can be made under severe amount of disease. Ground and aerial application are both acceptable."
        ]
    },
    {
        "name": "Corn(Maize)",
        "disease": "Common Rust",
        "oveview": [
            "Common rust is caused by the fungus Puccinia sorghi. Late occurring infections have limited impact on yield. The fungus overwinters on plants in southern states and airborne spores are wind-blown to northern states during the growing season. Disease development is favored by cool, moist weather (60 – 70◦ F).",
        "Symptoms of common rust often appear after silking. Small, round to elongate brown pustules form on both leaf surfaces and other above ground parts of the plant. As the pustules mature they become brown to black. If disease is severe, the leaves may yellow and die early.",
        "https://fieldcrops.cals.cornell.edu/corn/diseases-corn/common-rust"
        ],
        "solution":[
            "Commonly used fungicides include Aproach®, Headline®, Headline SC, Headline AMP®, PropiMax® EC, Quadris®, Quilt®, Quilt Xcel®, Stratego®, Stratego® YLD and Tilt®",
            "Monitor disease development, crop growth stage and weather forecast"
        ]
    },
    {
        "name": "Corn(Maize)",
        "disease": "Northern Leaf Blight",
        "oveview": [
            "Northern corn leaf blight caused by the fungus Exerohilum turcicum is a common leaf blight. If lesions begin early (before silking), crop loss can result. Late infections may have less of an impact on yield. Northern corn leaf blight is favored by wet humid cool weather typically found later in the growing season. Spores of the fungus that causes this disease can be transported by wind long distances from infected fields. Spread within and between fields locally also relies on wind blown spores.",
        "The tan lesions of northern corn leaf blight are slender and oblong tapering at the ends ranging in size between 1 to 6 inches. Lesions run parallel to the leaf margins beginning on the lower leaves and moving up the plant. They may coalesce and cover the enter leaf. Spores are produced on the underside of the leaf below the lesions giving the appearance of a dusty green fuzz.",
        "https://fieldcrops.cals.cornell.edu/corn/diseases-corn/northern-corn-leaf-blight"
        ],
        "solution":[
            "Northern corn leaf blight can be managed through the use of resistant hybrids.",
            "Additionally, timely planting can be useful for avoiding conditions that favor the disease."
        ]
    }, 
    {
        "name": "Corn(Maize)",
        "disease": "Healthy",
        "oveview": [
            "Healthy Corn"
        ],
        "solution":[
            "Your Corn are healthy. You took good care of it.",
            "Just take care of it as you usually do."
        ]
    },
    {
        "name": "Grape",
        "disease": "Black Rot",
        "oveview": [
            "Black rot is one of the most damaging diseases of grapes. The disease is caused by the fungus Guignardia bidwellii. The fungus can infect the leaves, shoots, berries, tendrils, rachises and cluster stems (peduncles) of grapes. If the disease is not managed early in the season, the impact on grape clusters can be devastating, resulting in complete crop losses.",
        "Disease development is favored by warm and humid weather. Symptoms of black rot first appear as small yellow spots on leaves. Enlarged spots (lesions) have a dark brownish-red border with tan to dark brown centers. As the infection develops, tiny black dots appear in the lesion, usually in a ring pattern near the border of the lesion. These dots are fungal structures (pycnidia), which contain thousands of spores (conidia) that can infect new tissue. New infections can occur in less than 10 hours at temperatures between 60 to 85 degrees Fahrenheit.",
        "https://ohioline.osu.edu/factsheet/plpath-fru-24"
        ],
        "solution":[
            "Look for characteristic dark circles on the leaves, dark raisin-like berries and lesions on the leaves and shoots of the grapes to confirm it is black rot. Before you can begin curing the rot, you must diagnose that what you are dealing with is in fact black rot.",
            "Cut off the obviously affected parts of the grape vine with a sterile knife. Remove all spotted leaves and the black, mummified grapes. Be extremely thorough and make sure you remove all parts of the plant that are affected by the black rot.",
            "Place fans in the growing area to keep the plants dry. This will help prevent the rot from spreading to other sections of the plant and will help dry out the newly cut parts of the plant. Black rot is quite contagious and is easily spread by water dripping from one affected part of the plant to another. Drying the plants as quickly as possible after watering or rainfall helps keep the disease in check.",
            "Apply a paste of cinnamon mixed with cooking oil to the remaining tissues of the plant. This paste -- when applied to the newly cut sections of the plant -- seals the wound and keeps it dry.",
            "Apply a fungicide, such as myclobutanil or captan according to the directions on the label.",
            "Re-apply the fungicide at two-week intervals until the black rot has been cured. In the meantime, carefully examine the plant and use a sterile knife to remove any additional affected areas. Apply the fungicide paste to the new cuts to help them heal."
        ]
    },
    {
        "name": "Grape",
        "disease": "Black Measles (Esca)",
        "oveview": [
            "Grapevine measles, also called esca, black measles or Spanish measles, has long plagued grape growers with its cryptic expression of symptoms and, for a long time, a lack of identifiable causal organism(s). The name \"measles\" refers to the superficial spots found on the fruit. During the season, the spots may coalesce over the skin surface, making berries black in appearance. Spotting can develop anytime between fruit set and a few days prior to harvest.",
        "Leaf symptoms are characterized by a \"tiger stripe\" pattern when infections are severe from year to year. Mild infections can produce leaf symptoms that can be confused with other diseases or nutritional deficiencies. White cultivars will display areas of chlorosis followed by necrosis, while red cultivars are characterized by red areas followed by necrosis. Early spring symptoms include shoot tip dieback, leaf discoloration and complete defoliation in severe cases.",
        "https://grapes.extension.org/grapevine-measles"
        ],
        "solution":[
            "THIOPHANATE-METHYL (Topsin-M WSB) Can be applied as a paint or spray application. Use allowed under a Special Local Needs label.",
            "MYCLOBUTANIL (Rally 40WSP) 4–5 oz Do not apply more than 1.5 lb of product/acre per season.",
            "TETRACONAZOLE (Mettle 125ME)3–5 oz  Do not apply more than 10 fl oz/acre per season."
        ]
    },
    {
        "name": "Grape",
        "disease": "Leaf Blight (Isariopsis Leaf Spot)",
        "oveview": [
            "Common in tropical and subtropical grapes. The disease appear late in the season. Cynthiana and Cabernet Sauvignon are susceptible to this pathogen.",
        "On leaf surface we will see lesions which are irregularly shaped (2 to 25 mm in diameter). Initially lesions are dull red to brown in color turn black later. If disease is severe this lesions may coalesce. On berries we can see symptom similar to black rot but the entire clusters will collapse.",
        "https://plantvillage.psu.edu/topics/grape/infos"
        ],
        "solution":[
            "a",
            "b",
            "c",
            "d"
        ]
    },
    {
        "name": "Grape",
        "disease": "Healthy",
        "oveview": [
            "Healthy Corn"
        ],
        "solution":[
            "Your Corn are healthy. You took good care of it.",
            "Just take care of it as you usually do."
        ]
    },
    {
        "name": "Orange",
        "disease": "Huanglongbing (Citrus Greening)",
        "oveview": [
            "Huanglongbing (HLB) or citrus greening is the most severe citrus disease, currently devastating the citrus industry worldwide. The presumed causal bacterial agent Candidatus Liberibacter spp. affects tree health as well as fruit development, ripening and quality of citrus fruits and juice. Fruit from infected orange trees can be either symptomatic or asymptomatic. Symptomatic oranges are small, asymmetrical and greener than healthy fruit. Furthermore, symptomatic oranges show higher titratable acidity and lower soluble solids, solids/acids ratio, total sugars, and malic acid levels.",
        "In the early stages of the disease, it is difficult to make a clear diagnosis. McCollum and Baldwin (2017) noted that HLB symptoms are more apparent during cooler seasons, more so than in warmer months. It is uncertain how long a tree can be infected before showing the symptoms of the disease but, when it eventually becomes symptomatic, symptoms appear on different parts of the tree. Infected trees generally develop some canopy thinning, with twig dieback and discolored leaves, which appear in contrast to the other healthy or symptomless parts of the tree.",
        "https://www.frontiersin.org/articles/10.3389/fpls.2018.01976/full"
        ],
        "solution":[
            "a",
            "b",
            "c",
            "d"
        ]
    },
    {
        "name": "Peach",
        "disease": "Bacterial spot",
        "oveview": [
            "Bacterial spot affects peaches, nectarines, apricots, plums, prunes and cherries. The disease is widespread throughout all fruit growing states east of the Rocky Mountains. Bacterial spot can affect leaves, twigs, and fruit. Severe infection results in reduced fruit quality and yield. Fruit infection is most serious on late-maturing varieties. If proper environmental conditions occur, up to 50 percent or more of the fruit of susceptible varieties may have to be discarded.",
        "Small (1/25 to 1/5 inch) spots form in the leaves. Spots are irregular to angular and have a deep purple to rusty-brown or black color. In time, the centers dry and tear away leaving ragged \"shot-holes\". When several spots merge, the leaf may appear scorched, blighted or ragged. Badly infected leaves may turn yellow and drop early. Early defoliation is most common on trees deficient in nitrogen or where the disease is further complicated by pesticide injury.",
        "https://ohioline.osu.edu/factsheet/plpath-fru-38"
        ],
        "solution":[
            "a",
            "b",
            "c",
            "d"
        ]
    },
    {
        "name": "Peach",
        "disease": "Healthy",
        "oveview": [
            "Healthy Corn"
        ],
        "solution":[
            "Your Corn are healthy. You took good care of it.",
            "Just take care of it as you usually do."
        ]
    },
    {
        "name": "Bell Pepper",
        "disease": "Bacterial Spot",
        "oveview": [
            "Bacterial leaf spot, caused by Xanthomonas campestris pv. vesicatoria, is the most common and destructive disease for peppers in the eastern United States. It is a  gram-negative, rod-shaped bacterium that can survive in seeds and plant debris from one season to another. Different strains or races of the bacterium are cultivar-specific, causing disease symptoms in certain varieties due to stringent host specificity. Bacterial leaf spot can devastate a pepper crop by early defoliation of infected leaves and disfiguring fruit.",
        "Disease symptoms can appear throughout the above-ground portion of the plant, which may include leaf spot, fruit spot and stem canker. However, early symptoms show up as water-soaked lesions on leaves that can quickly change from green to dark brown and enlarge into spots that are up to 1/4 inch in diameter with slightly raised margins. Over time, these spots can dry up in less humid weather, which allows the damaged tissues to fall off, resulting in a tattered appearance on the affected leaves.",
        "https://extension.wvu.edu/lawn-gardening-pests/plant-disease/fruit-vegetable-diseases/bacterial-leaf-spot-of-pepper"
        ],
        "solution":[
            "a",
            "b",
            "c",
            "d"
        ]
    },
    {
        "name": "Bell Pepper",
        "disease": "Healthy",
        "oveview": [
            "Healthy Corn"
        ],
        "solution":[
            "Your Corn are healthy. You took good care of it.",
            "Just take care of it as you usually do."
        ]
    },
    {
        "name": "Potato",
        "disease": "Early Blight",
        "oveview": [
            "Common on tomato and potato plants, early blight is caused by the fungus Alternaria solani. Symptoms first appear on the lower, older leaves as small brown spots with concentric rings that form a \"bull’s eye\" pattern. As the disease matures, it spreads outward on the leaf surface causing it to turn yellow, wither and die. Eventually the stem, fruit and upper portion of the plant will become infected. Crops can be severely damaged.",
        "Early blight overwinters on infected plant tissue and is spread by splashing rain, irrigation, insects and garden tools. The disease is also carried on tomato seeds and in potato tubers. In spite of its name, early blight can occur any time throughout the growing season. High temperatures (80-85˚F.) and wet, humid conditions promote its rapid spread. In many cases, poorly nourished or stressed plants are attacked.",
        "https://www.planetnatural.com/pest-problem-solver/plant-disease/early-blight"
        ],
        "solution":[
            "Prune or stake plants to improve air circulation and reduce fungal problems.",
            "Make sure to disinfect your pruning shears (one part bleach to 4 parts water) after each cut.",
            "Keep the soil under plants clean and free of garden debris. Add a layer of organic compost to prevent the spores from splashing back up onto vegetation.",
            "Drip irrigation and soaker hoses can be used to help keep the foliage dry.",
            "For best control, apply copper-based fungicides early, two weeks before disease normally appears or when weather forecasts predict a long period of wet weather. Alternatively, begin treatment when disease first appears, and repeat every 7-10 days for as long as needed.",
            "Containing copper and pyrethrins, Bonide® Garden Dust is a safe, one-step control for many insect attacks and fungal problems. For best results, cover both the tops and undersides of leaves with a thin uniform film or dust. Depending on foliage density, 10 oz will cover 625 sq ft. Repeat applications every 7-10 days, as needed."
        ]
    },
    {
        "name": "Potato",
        "disease": "Late Blight",
        "oveview": [
            "Late blight (Phytophthora infestans) fungus is in the same genus as the fungus causing pink rot (P. erythroseptica). Late blight was responsible for the Irish potato famine in the mid-nineteenth century (Daly, 1996). In the late twentieth century, there have been major re-occurrences and concern around the world over this pathogen and its disease due to recent mutations (Fry and Goodwin, 1997). These mutations, most notably strain US-8, have made the pathogen resistant to control by metalaxyl, the stand-by fungicide for many years.",
        "Late blight will first appear as water-soaked spots, usually at the tips or edges of lower leaves where water or dew tends to collect. Under moist, cool conditions, water-soaked spots rapidly enlarge and a broad yellow halo may be seen surrounding the lesion (Mohan et al., 1996). On the leaf underside, a spore-producing zone of white moldy growth approximately 0.1 - 0.2 inches wide may appear at the border of the lesion. Under continuously wet conditions, the disease progresses rapidly and warm, dry weather will slow or stop disease development.",
        "https://cropwatch.unl.edu/potato/late_blights"
        ],
        "solution":[
            "Plant resistant cultivars when available.",
            "Remove volunteers from the garden prior to planting and space plants far enough apart to allow for plenty of air circulation.",
            "Water in the early morning hours, or use soaker hoses, to give plants time to dry out during the day — avoid overhead irrigation.",
            "Destroy all tomato and potato debris after harvest (see Fall Garden Cleanup)."
        ]
    },
    {
        "name": "Potato",
        "disease": "Healthy",
        "oveview": [
            "Healthy Potato"
        ],
        "solution":[
            "Your Potato are healthy. You took good care of it.",
            "Just take care of it as you usually do."
        ]
    },
    {
        "name": "Raspberry",
        "disease": "Healthy",
        "oveview": [
            "Healthy Corn"
        ],
        "solution":[
            "Your Corn are healthy. You took good care of it.",
            "Just take care of it as you usually do."
        ]
    },
    {
        "name": "Rice",
        "disease": "Brown Spot",
        "oveview": [
            "Brown Spot is called as sesame leaf spot or Helminthosporiose or fungal blight. The fungus attacks the crop from seedling in nursery to milk stage in main field.",
        "The disease appears first as minute brown dots, later becoming cylindrical or oval to circular (resemble sesame seed). Spots measures 0.5 to 2.0mm in breadth - coalesce to form large patches. Then several spots coalesce and the leaf dries up. Infection also occurs on panicle, neck with brown colour appearance. Seeds also infected (black or brown spots on glumes spots are covered by olivaceous velvety growth)",
        "http://www.agritech.tnau.ac.in/expert_system/paddy/cpdisbrownspot.html"
        ],
        "solution":[
            "a",
            "b",
            "c",
            "d"
        ]
    },
    {
        "name": "Rice",
        "disease": "Hispa",
        "oveview": [
            "The mining of the grubs will be clearly seen on the leaves. Scraping of the upper surface of the leaf blade leaving only the lower epidermis as white streaks parallel to the midrib. Tunneling of larvae through leaf tissue causes irregular translucent white patches that are parallel to the leaf veins. Damaged leaves wither off. Rice field appears burnt when severely infested.",
        "The grub mines into the leaf blade and feed on the green tissue between the veins. Adults also feed in the green tissue; they scrape green matter of the tender leaves. Generally the plants are affected in the young stage.",
        "http://www.agritech.tnau.ac.in/expert_system/paddy/cppests_ricehispa.html"
        ],
        "solution":[
            "a",
            "b",
            "c",
            "d"
        ]
    },
    {
        "name": "Rice",
        "disease": "Leaf Blast",
        "oveview": [
            "Blast, also called rotten neck, is one of the most destructive diseases of Missouri rice. Losses due to this disease have been on the increase since 2000. Blast does not develop every year but is very destructive when it occurs. Rice blast can be controlled by a combination of preventive measures and foliar fungicides applied when rice is in the late boot stage and again when it is 80 to 90 percent headed.",
        "Blast symptoms can occur on leaves, leaf collars, nodes and panicles. Leaf spots are typically elliptical (football shaped), with gray-white centers and brown to red-brown margins. Fully developed leaf lesions are approximately 0.4 to 0.7 inch long and 0.1 to 0.2 inch wide. Both the shape and the color vary depending on the environment, age of the lesion and rice variety. Lesions on leaf sheaths, which rarely develop, resemble those on leaves.",
        "https://extension.missouri.edu/publications/mp645"
        ],
        "solution":[
            "a",
            "b",
            "c",
            "d"
        ]
    },
    {
        "name": "Rice",
        "disease": "Healthy",
        "oveview": [
            "Healthy Corn"
        ],
        "solution":[
            "Your Corn are healthy. You took good care of it.",
            "Just take care of it as you usually do."
        ]
    },
    {
        "name": "Soybean",
        "disease": "Healthy",
        "oveview": [
            "Healthy Corn"
        ],
        "solution":[
            "Your Corn are healthy. You took good care of it.",
            "Just take care of it as you usually do."
        ]
    },
    {
        "name": "Squash",
        "disease": "Powdery Mildew",
        "oveview": [
            "Powdery mildew, mainly caused by the fungus Podosphaera xanthii, infects all cucurbits, including muskmelons, squash, cucumbers, gourds, watermelons and pumpkins. In severe cases, powdery mildew can cause premature death of leaves, and reduce yield and fruit quality.",
        "The first sign of powdery mildew is pale yellow leaf spots. White powdery spots can form on both upper and lower leaf surfaces, and quickly expand into large blotches. The large blotches can cover entire leaf, petiole and stem surfaces. When powdery mildew infects the majority of the foliage, the plant weakens and the fruit ripens prematurely.",
        "https://extension.umn.edu/diseases/powdery-mildew-cucurbits"
        ],
        "solution":[
            "a",
            "b",
            "c",
            "d"
        ]
    },
    {
        "name": "Strawberry",
        "disease": "Leaf Scorch",
        "oveview": [
            "In addition to leaves, leaf scorch (Diplocarpon earlianum) can infect petioles, runners, fruit stalks and berry caps. If unchecked, plants can be significantly weakened reducing the growth of all plant parts. Severely infected plants are weakened and can die from other stresses such as drought or extreme temperatures.",
        "Dark purple, angular to round spots appear on the upper surface of the leaf. As the disease progresses the tissues around these spots turn reddish or purple. In severe cases, the infected area dries to a tan color and the leaf curls upward looking scorched. Lesions remain reddish purple and do not turn tan or gray in the center.",
        "https://extension.umn.edu/fruit/growing-strawberries-home-garden#gray-mold%2C-leaf-blight%2C-leaf-scorch-and-leaf-spot--1008160"
        ],
        "solution":[
            "a",
            "b",
            "c",
            "d"
        ]
    },
    {
        "name": "Strawberry",
        "disease": "Healthy",
        "oveview": [
            "Healthy Corn"
        ],
        "solution":[
            "Your Corn are healthy. You took good care of it.",
            "Just take care of it as you usually do."
        ]
    },
    {
        "name": "Tomato",
        "disease": "Bacterial Spot",
        "oveview": [
            "Bacterial spot can be a devastating disease when the weather is warm and humid. The disease can affect all above-ground parts of tomato and pepper plants: stems, petioles, leaves, and fruits. Fruit spots commonly result in unmarketable fruit, not only for fresh market but also for processing because the spots make the fruit difficult to peel.",
        "Tomato leaves have small (<1/8 inch), brown, circular spots surrounded by a yellow halo. The center of the leaf spots often falls out resulting in small holes. Small, brown, circular spots may also occur on stems and the fruit calyx. Fruit spots are ¼ inch, slightly raised, brown and scabby. Tomato fruit often have a waxy white halo surrounding the fruit spot.",
        "https://extension.umn.edu/diseases/bacterial-spot-tomato-and-pepper"
        ],
        "solution":[
            "a",
            "b",
            "c",
            "d"
        ]
    },
    {
        "name": "Tomato",
        "disease": "Early Blight",
        "oveview": [
            "Early blight is one of the most common tomato diseases, occurring nearly every season wherever tomatoes are grown. It affects leaves, fruits and stems and can be severely yield limiting when susceptible cultivars are used and weather is favorable. Severe defoliation can occur and result in sunscald on the fruit. Early blight is common in both field and high tunnel tomato production in Minnesota.",
        "Initially, small dark spots form on older foliage near the ground. Leaf spots are round, brown and can grow up to half inch in diameter. Larger spots have target-like concentric rings. The tissue around spots often turns yellow. Severely infected leaves turn brown and fall off, or dead, dried leaves may cling to the stem.",
        "https://extension.umn.edu/diseases/early-blight-tomato"
        ],
        "solution":[
            "a",
            "b",
            "c",
            "d"
        ]
    },
    {
        "name": "Tomato",
        "disease": "Late Blight",
        "oveview": [
            "Late blight is a potentially devastating disease of tomato and potato, infecting leaves, stems and fruits of tomato plants. The disease spreads quickly in fields and can result in total crop failure if untreated. Late blight of potato was responsible for the Irish potato famine of the late 1840s.",
        "Leaves have large, dark brown blotches with a green gray edge; not confined by major leaf veins. Infections progress through leaflets and petioles, resulting in large sections of dry brown foliage. Stem infections are firm and dark brown with a rounded edge.",
        "https://extension.umn.edu/diseases/late-blight"
        ],
        "solution":[
            "a",
            "b",
            "c",
            "d"
        ]
    },
    {
        "name": "Tomato",
        "disease": "Leaf Mold",
        "oveview": [
            "Leaf mold is not normally a problem in field-grown tomatoes in northern climates. It can cause losses in tomatoes grown in greenhouses or high tunnels due to the higher humidity found in these environments. Foliage is often the only part of the plant infected and will cause infected leaves to wither and die, indirectly affecting yield. In severe cases, blossoms and fruit can also be infected, directly reducing yield.",
            "The oldest leaves are infected first. Pale greenish-yellow spots, usually less than 1/4 inch, with no definite margins, form on upper sides of leaves. Olive-green to brown velvety mold forms on the lower leaf surface below leaf spots. Leaf spots grow together and turn brown. Leaves wither and die but often remain attached to the plant.",
            "https://extension.umn.edu/diseases/leaf-mold-tomato"
        ],
        "solution":[
            "a",
            "b",
            "c",
            "d"
        ]
    },
    {
        "name": "Tomato",
        "disease": "Septoria Leaf Spot",
        "oveview": [
            "Septoria leaf spot is a very common disease of tomatoes. It is caused by a fungus (Septoria lycopersici) and can affect tomatoes and other plants in the Solanaceae family, especially potatoes and eggplant, just about anywhere in the world. Although Septoria leaf spot is not necessarily fatal for your tomato plants, it spreads rapidly and can quickly defoliate and weaken the plants, rendering them unable to bear fruit to maturity.",
        "Septoria leaf spots start off somewhat circular and first appear on the undersides of older leaves, at the bottom of the plant. They are small, 1/16 to 1/8 inches (1.6 to 3.2 millimeters) in diameter, with a dark brown margin and lighter gray or tan centers. A yellow halo may surround the spot.",
        "https://www.thespruce.com/identifying-and-controlling-septoria-leaf-spot-of-tomato-1402974"
        ],
        "solution":[
            "a",
            "b",
            "c",
            "d"
        ]
    },
    {
        "name": "Tomato",
        "disease": "Spider Mites (Two-spotted Spider Mite)",
        "oveview": [
            "Many species of the spider mite (family: Tetranychidae), so common in North America, attack both indoor and outdoor plants. They can be especially destructive in greenhouses. Spider mites are not true insects, but are classed as a type of arachnid, relatives of spiders, ticks and scorpions.",
        "Spider mites, almost too small to be seen, pass into our gardens without notice. No matter how few, each survives by sucking material from plant cells. Large infestations cause visible damage. Leaves first show patterns of tiny spots or stipplings. They may change color, curl and fall off. The mites activity is visible in the tight webs that are formed under leaves and along stems.",
        "https://www.planetnatural.com/pest-problem-solver/houseplant-pests/spider-mite-control"
        ],
        "solution":[
            "a",
            "b",
            "c",
            "d"
        ]
    },
    {
        "name": "Tomato",
        "disease": "Target Spot",
        "oveview": [
            "Also known as early blight, target spot of tomato is a fungal disease that attacks a diverse assortment of plants, including papaya, peppers, snap beans, potatoes, cantaloupe, and squash as well as passion flower and certain ornamentals. Target spot on tomato fruit is difficult to control because the spores, which survive on plant refuse in the soil, are carried over from season to season.",
        "Target spot on tomato fruit is difficult to recognize in the early stages, as the disease resembles several other fungal diseases of tomatoes. However, as diseased tomatoes ripen and turn from green to red, the fruit displays circular spots with concentric, target-like rings and a velvety black, fungal lesions in the center. The \"targets\" become pitted and larger as the tomato matures.",
        "https://www.gardeningknowhow.com/edible/vegetables/tomato/target-spot-on-tomatoes.htm"
        ],
        "solution":[
            "a",
            "b",
            "c",
            "d"
        ]
    },
    {
        "name": "Tomato",
        "disease": "Yellow Leaf Curl Virus",
        "oveview": [
            "Tomato yellow leaf curl virus is undoubtedly one of the most damaging pathogens of tomato, and it limits production of tomato in many tropical and subtropical areas of the world. It is also a problem in many countries that have a Mediterranean climate such as California. Thus, the spread of the virus throughout California must be considered as a serious potential threat to the tomato industry.",
            "Infected tomato plants initially show stunted and erect or upright plant growth; plants infected at an early stage of growth will show severe stunting. However, the most diagnostic symptoms are those in leaves.",
            "https://www2.ipm.ucanr.edu/agriculture/tomato/tomato-yellow-leaf-curl"
        ],
        "solution":[
            "a",
            "b",
            "c",
            "d"
        ]
    },
    {
        "name": "Tomato",
        "disease": "Mosaic Virus",
        "oveview": [
            "Tomato mosaic virus (ToMV) and  Tobacco mosaic virus (TMV) are hard to distinguish. Tomato mosaic virus (ToMV) can cause yellowing and stunting of tomato plants resulting in loss of stand and reduced yield. ToMV may cause uneven ripening of fruit, further reducing yield.",
        "Mottled light and dark green on leaves. If plants are infected early, they may appear yellow and stunted overall. Leaves may be curled, malformed, or reduced in size. Spots of dead leaf tissue may become apparent with certain cultivars at warm temperatures. Fruits may ripen unevenly. Reduced fruit number and size.",
        "https://extension.umn.edu/diseases/tomato-mosaic-virus-and-tobacco-mosaic-virus"
        ],
        "solution":[
            "a",
            "b",
            "c",
            "d"
        ]
    },
    {
        "name": "Tomato",
        "disease": "Healthy",
        "oveview": [
            "Healthy Corn"
        ],
        "solution":[
            "Your Corn are healthy. You took good care of it.",
            "Just take care of it as you usually do."
        ]
    },
    {
        "name": "chili",
        "disease": "Healthy",
        "oveview": [
            "Healthy Corn"
        ],
        "solution":[
            "Your Corn are healthy. You took good care of it.",
            "Just take care of it as you usually do."
        ]
    },
    {
        "name": "chili",
        "disease": "leaf curl",
        "oveview": [
            "chili leaf curl (Taphrina deformans) is a common fungal disease of peaches, ornamental peaches, and nectarines. It has also been found in apricot trees and occasionally in almond trees. The fungus infects leaves, fruits, blossoms, and shoots. While chili leaf curl is serious, there are ways to protect your trees from it."
        ],
        "solution":[
            "Select resistant varieties whenever possible.",
            "Leaf curl can be controlled by applying sulfur or copper-based fungicides that are labeled for use on peaches and nectarines. Spray the entire tree after 90% of the leaves have dropped in the fall and again in the early spring, just before the buds open. For best results, trees should be sprayed to the point of runoff or until they start dripping.",
            "Containing copper and pyrethrins, Bonide® Garden Dust is a safe, one-step control for many insect attacks and fungal problems. For best results, cover both the tops and undersides of leaves with a thin uniform film or dust. Depending on foliage density, 10 oz will cover 625 sq ft. Repeat applications every 7-10 days, as needed.",
            "Keep the ground beneath the trees raked up and clean, especially during winter months.",
            "Prune and destroy infected plant parts as they appear.",
            "If disease problems are severe, maintain tree health and vigor by cutting back more fruit than normal, watering regularly (avoiding wetting the leaves if possible) and apply an organic fertilizers high in nitrogen."
        ]
    },
    {
        "name": "chili",
        "disease": "leaf spot",
        "oveview": [
            "One common issue we are asked about often are various types of spots on pepper leaves.",
            "If you see large, white or gray patches on your pepper leaves, it is almost certainly sun scald that is causing the problem. These can appear as small spots or large patches, especially on older leaves.",
            "Sunscald is basically a plant sun burn. Improper hardening off leads to this type of leaf burn on many types of plants, including peppers. The white spots are burned, dried out leaf tissue that is dead."
        ],
        "solution":[
            "When selecting fruit trees, choose resistant varieties if possible.",
            "Keep the soil under the tree clean and rake up fallen fruit.",
            "Use a thick layer of mulch to cover the soil after you have raked and cleaned it well. Mulch will reduce weeds and prevent the disease pathogen from splashing back up onto the leaves.",
            "Prune or stake plants to improve air circulation. Make sure to disinfect your pruning equipment (one part bleach to 4 parts water) after each cut.",
            "Leaf spot among vegetables is most often introduced through infected seed or transplants. Make sure your seeds and transplants are from leaf spot-free stock."
        ]
    },
    {
        "name": "chili",
        "disease": "whitefy",
        "oveview": [
            "Chilli is one of the most important commercial crop grown in India. India is the major producer, consumer and exporter of chilli in the world. The major chilli growing states in India are Andhra Pradesh, Karnataka, Maharashtra, Tamil Nadu, Bihar and U.P. these states accounts for nearly 80% area under chilli cultivation in India. Chilli can be grown in all types of soil from light sandy to heavy soils. Chilli is very sensitive to water logging. Chilli is growing under both irrigated and rainfed condition. Even though its a rainfed crop, farmers consider this as one of the commercial crop."
        ],
        "solution":[
            "In a liter of water, mix 5 ml of Verticillium, 5 ml of neem oil, 5 ml of castor oil and 5 g of powdered jaggery for about 5 minutes. Then spray the solution on the lower leaves of the plant with a sprayer. This means that whiteflies are more common on the lower leaves of the plant."
        ]
    },
    {
        "name": "chili",
        "disease": "yellowish",
        "oveview": [
            "Clearly, yellow leaves are a problem; it’s an indication that something is wrong and you need to figure it out pronto.",
            "Pepper plants are one of the most popular veggies in home gardens, partially because they can survive a lot of conditions. That doesn’t mean that they’re immune to problems. One issue you’ll often encounter is yellow leaves on your pepper plants."
        ],
        "solution":[
            "If you’re growing pepper plants in containers, consistent watering is crucial, along with picking a container that has plenty of drainage holes. Pepper plants don’t like waterlogged roots.",
            "One way you can figure out if a nitrogen deficiency is your problem is if the yellow leaves begin at the bottom of your plant, slowly moving upwards. It does this because nitrogen is called a mobile nutrient, which means it travels over time from one part of your plant to the next."

        ]
    }
]"""
    aList = json.loads(jsonStr)
    for obj in aList:
      if(obj["name"].lower() == name.lower() and obj["disease"].lower() == disease.lower()):
        return obj

