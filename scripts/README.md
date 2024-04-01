Experiments Scripts 
===================

In order to run experiments just select and run appropriate script. 
They have different arguments, but more or less in this args: 

```bash
./script_expXXX.sh <approach/dataset> <gpu> [<results_dir>]
```

where: 
    
- `<approach/dataset>` - script dependent, just check inside the script

- `<gpu>` - index of GPU to run on the experiment

- `[<results_dir>]` - results dir (optional), by defoult it will be `./results`
