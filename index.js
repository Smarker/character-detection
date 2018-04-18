var fs = require('fs-extra'),
    xml2js = require('xml2js'),
    _=require('lodash');

    const filenames = fs.readdirSync(__dirname+'/labels');
    var parser = new xml2js.Parser();
    filenames.forEach(eachfile=>{
        filedata = fs.readFileSync(__dirname+'/labels/'+eachfile);      
        parser.parseString(filedata,(err,result)=>{
            console.log(result);
            boxes = _.get(result,'annotation.object');
            res = _.map(boxes,eachbox=>{
                console.log(eachbox);
                let { xmin,ymin,xmax,ymax} = eachbox.bndbox[0];
                return `${xmin},${ymin},${xmin},${ymax},${xmax},${ymax},${xmax},${ymin}`
            })
            fs.writeFileSync(__dirname + '/newlabels/gt_' + eachfile.split('.xml')[0] + '.txt',res.join('\n'))
            parser.reset();
        });
    })